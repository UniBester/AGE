from argparse import ArgumentParser


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--dataset_type', default='af_encode', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use')
		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
		self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')

		self.parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
		self.parser.add_argument('--valid_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=2, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--valid_workers', default=2, type=int, help='Number of test/inference dataloader workers')
		self.parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

		self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
		self.parser.add_argument('--A_length', default=100, type=int, help='length of A')
		self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
		self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
		self.parser.add_argument('--start_from_latent_avg', action='store_true',
		                         help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space insteaf of w+')

		self.parser.add_argument('--class_embedding_path', default=None, type=str, help='path to class embedding')
		self.parser.add_argument('--psp_checkpoint_path', default=None, type=str, help='Path to pretrained pSp model checkpoint')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to AGE model checkpoint')

		self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--sparse_lambda', default=0.005, type=float, help='sparse loss for n')
		self.parser.add_argument('--orthogonal_lambda', default=0.0005, type=float, help='orthogonal loss multiplier factor for A')
		self.parser.add_argument('--lpips_lambda', default=1.0, type=float, help='LPIPS loss multiplier factor')

		self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=3000, type=int, help='Model checkpoint interval')


	def parse(self):
		opts = self.parser.parse_args()
		return opts