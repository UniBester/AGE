"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from models.encoders import psp_encoders
from models.stylegan2.model import Generator




class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim):
        super(EqualLinear, self).__init__()
        self.out_dim=out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim*3, in_dim*2)
        self.fc2 = nn.Linear(in_dim*2, in_dim*2)
        self.fc3 = nn.Linear(in_dim*2, in_dim)
        self.fc4 = nn.Linear(in_dim, out_dim)
        self.fc5 = nn.Linear(out_dim, out_dim)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        out = self.nonlinearity(out)
        out = self.fc4(out)
        out = self.nonlinearity(out)
        out = self.fc5(out)
        return out



class Ax(nn.Module):
	def __init__(self, dim):
		super(Ax, self).__init__()
		self.A=nn.Parameter(torch.randn(6, 512, dim), requires_grad=True)
		self.encoder0=EqualLinear(512, dim)
		self.encoder1=EqualLinear(512, dim)
	def forward(self, dw):
		x0=self.encoder0(dw[:, :3])
		x0=x0.unsqueeze(-1).unsqueeze(1)
		x1=self.encoder1(dw[:, 3:6])
		x1=x1.unsqueeze(-1).unsqueeze(1)
		x=[x0.squeeze(-1),x1.squeeze(-1)]
		output_dw0=torch.matmul(self.A[:3], x0).squeeze(-1)
		output_dw1=torch.matmul(self.A[3:6], x1).squeeze(-1)
		output_dw=torch.cat((output_dw0,output_dw1),dim=1)
		return output_dw, self.A, x



def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class AGE(nn.Module):

	def __init__(self, opts):
		super(AGE, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		self.ax = Ax(self.opts.A_length)
		self.decoder = Generator(self.opts.output_size, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			if dist.is_initialized():
				if dist.get_rank()==0:
					print('Loading AGE from checkpoint: {}'.format(self.opts.checkpoint_path))
			else:
				print('Loading AGE from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location=torch.device('cpu'))
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in ckpt['state_dict'].items():
				name = k.replace('module.','') 
				new_state_dict[name] = v
			ckpt['state_dict'] = new_state_dict
			self.ax.load_state_dict(get_keys(ckpt, 'ax'), strict=True)
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			if dist.is_initialized():
				if dist.get_rank()==0:
					print('Loading pSp from checkpoint: {}'.format(self.opts.psp_checkpoint_path))
			else:
				print('Loading pSp from checkpoint: {}'.format(self.opts.psp_checkpoint_path))
			ckpt = torch.load(self.opts.psp_checkpoint_path, map_location=torch.device('cpu'))
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in ckpt['state_dict'].items():
				name = k.replace('.module','') 
				new_state_dict[name] = v
			ckpt['state_dict'] = new_state_dict
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)

	def forward(self, x, av_codes, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			ocodes = self.encoder(x)
			odw = ocodes[:, :6] - av_codes[:, :6]
			dw, A, x = self.ax(odw)
			codes = torch.cat((dw + av_codes[:, :6], ocodes[:, 6:]), dim=1)

			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		#image generation
		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return {'y_hat':images, 'latent':result_latent, 'dw':[odw, dw], 'A':A, 'x':x}
		else:
			return {'y_hat':images, 'dw':[odw, dw], 'A':A, 'x':x}

		# return {'dw':[odw, dw], 'A':A}

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None

	def get_code(self, x, av_codes, resize=True, latent_mask=None, return_latents=False):
		ocodes = self.encoder(x)
		odw = ocodes - av_codes
		dw, A, x = self.ax(odw)
		codes = torch.cat((dw + av_codes[:, :6], ocodes[:, 6:]), dim=1)
		return {'odw':odw, 'dw':dw, 'A':A, 'x':x, 'codes':codes, 'ocodes':ocodes}

	def get_test_code(self, x, resize=True, latent_mask=None, return_latents=False):
		ocodes = self.encoder(x)
		odw = ocodes[:, :6]
		dw, A, x = self.ax(odw)
		return { 'A':A,  'ocodes':ocodes}
	
	def decode(self, codes, resize=True, input_code=False, randomize_noise=True, return_latents=False):
		# normalize with respect to the center of an average face
		if self.opts.start_from_latent_avg:
			if self.opts.learn_in_w:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
			else:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		#image generation
		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		return images