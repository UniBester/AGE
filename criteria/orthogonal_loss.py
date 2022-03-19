import torch
from torch import nn


class OrthogonalLoss(nn.Module):

	def __init__(self, opts):
		super(OrthogonalLoss, self).__init__()
		# self.device = 'cuda'  # elodie
		B_path = opts.class_embedding_path
		self.B = torch.stack(list(torch.load(B_path).values())).to(opts.device).permute(1,2,0)[:6]

	def forward(self, A):
		dim = A.shape[-1]
		return torch.sum(torch.abs(torch.matmul(A.transpose(1,2), self.B)))/(dim*self.B.shape[-1]*6)
