import os
import json
import pprint
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler



from utils import common, train_utils
from criteria import orthogonal_loss, sparse_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from models.age import AGE
from training.ranger import Ranger


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		dist.init_process_group(backend="nccl")
		self.local_rank = dist.get_rank()
		torch.cuda.set_device(self.local_rank)
		self.device = torch.device("cuda", self.local_rank)
		self.opts.device = self.device

		# Initialize network
		self.net = AGE(self.opts).to(self.device)
		# self.net.encoder = nn.DataParallel(self.net.encoder)
		# self.net.decoder = nn.DataParallel(self.net.decoder)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

		self.net = nn.parallel.DistributedDataParallel(self.net, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

		# Initialize loss
		if self.opts.sparse_lambda > 0:
			self.sparse_loss = sparse_loss.SparseLoss().to(self.device).eval()
		if self.opts.orthogonal_lambda > 0:
			self.orthogonal_loss = orthogonal_loss.OrthogonalLoss(self.opts).to(self.device).eval()
			B_path = self.opts.class_embedding_path
			self.B = torch.stack(list(torch.load(B_path).values())).to(self.device).permute(1,2,0)[:6]

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.valid_dataset = self.configure_datasets()
		self.train_sampler = DistributedSampler(self.train_dataset)
		self.valid_dsampler = DistributedSampler(self.valid_dataset, shuffle=False)
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   num_workers=int(self.opts.workers),
										   sampler=self.train_sampler)
		self.valid_dataloader = DataLoader(self.valid_dataset,
										  batch_size=self.opts.valid_batch_size,
										  num_workers=int(self.opts.valid_workers),
										  sampler=self.valid_dsampler)

		if self.local_rank==0:
			# Initialize logger
			if os.path.exists(opts.exp_dir):
				raise Exception('Oops... {} already exists'.format(opts.exp_dir))
			os.makedirs(opts.exp_dir)

			# opts_dict = vars(opts)
			# pprint.pprint(opts_dict)
			# with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
			# 	json.dump(opts_dict, f, indent=4, sort_keys=True)

			log_dir = os.path.join(opts.exp_dir, 'logs')
			os.makedirs(log_dir, exist_ok=True)
			self.logger = SummaryWriter(log_dir=log_dir)

			# Initialize checkpoint dir
			self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
			os.makedirs(self.checkpoint_dir, exist_ok=True)
			self.best_val_loss = None
			if self.opts.save_interval is None:
				self.opts.save_interval = self.opts.max_steps

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()
				x, y, av_codes = batch
				x, y, av_codes = x.to(self.device).float(), y.to(self.device).float(), av_codes.to(self.device).float()
				outputs = self.net.forward(x, av_codes, return_latents=True)
				loss, loss_dict, id_logs = self.calc_loss(outputs, y)
				loss.backward()
				self.optimizer.step()

				# Logging related
				if self.local_rank==0:
					if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
						self.parse_and_log_images(id_logs, x, y, outputs['y_hat'], title='images/train/faces')
					if self.global_step % self.opts.board_interval == 0:
						self.print_metrics(loss_dict, prefix='train')
						self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if self.local_rank==0 and val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)
				if self.local_rank==0:
					if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
						if val_loss_dict is not None:
							self.checkpoint_me(val_loss_dict, is_best=False)
						else:
							self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					if self.local_rank==0:
						print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.valid_dataloader):
			x, y, av_codes = batch
			with torch.no_grad():
				x, y, av_codes = x.to(self.device).float(), y.to(self.device).float(), av_codes.to(self.device).float()
				outputs = self.net.forward(x, av_codes, return_latents=True)
				loss, cur_loss_dict, id_logs = self.calc_loss(outputs, y)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			if self.local_rank==0:
				self.parse_and_log_images(id_logs, x, y, outputs['y_hat'], title='images/valid/faces', subscript='{:04d}'.format(batch_idx))

			# For first step just do sanity valid on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		self.net.train()

		if self.local_rank==0:
			loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
			self.log_metrics(loss_dict, prefix='valid')
			self.print_metrics(loss_dict, prefix='valid')
			return loss_dict

		return None

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		params = list(self.net.module.ax.parameters())
		if self.opts.train_decoder:
			params += list(self.net.module.decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		if self.local_rank==0:
			print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  average_code_root=self.opts.class_embedding_path,
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		valid_dataset = ImagesDataset(source_root=dataset_args['valid_source_root'],
									 target_root=dataset_args['valid_target_root'],
									 average_code_root=self.opts.class_embedding_path,
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_valid'],
									 opts=self.opts)
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of valid samples: {len(valid_dataset)}")
		return train_dataset, valid_dataset

	def calc_loss(self, outputs, y):
		loss_dict = {}
		loss = 0.0
		id_logs = None
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(outputs['y_hat'], y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.orthogonal_lambda > 0:
			loss_orthogonal_AB = self.orthogonal_loss(outputs['A'])
			loss_dict['loss_orthogonal_AB'] = float(loss_orthogonal_AB)
			loss += (loss_orthogonal_AB) * self.opts.orthogonal_lambda
		if self.opts.sparse_lambda > 0:
			loss_l1 = self.sparse_loss(outputs['x'])
			loss_dict['loss_l1'] = float(loss_l1)
			loss += loss_l1 * self.opts.sparse_lambda
		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.log_input_image(x[i], self.opts),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
		else:
			path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.net.module.latent_avg
		return save_dict
