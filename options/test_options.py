from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--dataset_type', default='fl_encode', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--psp_checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to AGE model checkpoint')
		self.parser.add_argument('--n_distribution_path', type=str, default=None, help='Path to distribution of n')
		self.parser.add_argument('--train_data_path', type=str, default=None, help='Path to directory of training set')
		self.parser.add_argument('--test_data_path', type=str, default=None, help='Path to to directory of inference inputs')
		self.parser.add_argument('--output_path', type=str, default=None, help='Path to save outputs')
		self.parser.add_argument('--class_embedding_path', type=str, default=None, help='Path to save class embeddings')
		
		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')
		
		self.parser.add_argument('--n_images', type=int, default=128, help='Number of images to generate on per input')
		self.parser.add_argument('--A_length', default=100, type=int, help='Length of A')
		self.parser.add_argument('--alpha', default=1, type=float, help='Editing intensity alpha')
		self.parser.add_argument('--beta', default=0.000, type=float, help='Direction selection threshold in A')
		self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')
	def parse(self):
		opts = self.parser.parse_args()
		return opts