"""
This file runs the main training/val loop
"""
import os
import sys
import torch

sys.path.append(".")
sys.path.append("..")
os.environ["OMP_NUM_THREADS"]="1"

from options.train_options import TrainOptions
from training.coach import Coach

	

if __name__ == '__main__':
	torch.multiprocessing.set_start_method('spawn')
	opts = TrainOptions().parse()
	coach = Coach(opts)
	coach.train()

