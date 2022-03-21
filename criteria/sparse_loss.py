import torch
from torch import nn

class SparseLoss(nn.Module):

	def __init__(self):
		super(SparseLoss, self).__init__()
		self.theta0=0.5
		self.theta1=-1

	def forward(self, X):
		x0 = torch.sigmoid(self.theta0*X[0].abs()+self.theta1)
		x1 = torch.sigmoid(self.theta0*X[1].abs()+self.theta1)
			
		return x0.sum()/32+x1.sum()/32
