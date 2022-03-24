import os
from argparse import Namespace

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys
import json

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.age import AGE




def run():
    test_opts = TestOptions().parse()
    ckpt = torch.load(test_opts.psp_checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = AGE(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    data_path=test_opts.train_data_path
    class_embedding_path=test_opts.class_embedding_path
    os.makedirs(class_embedding_path, exist_ok=True)
    dataset = InferenceDataset(root=data_path,
                            transform=transforms_dict['transform_inference'],
                            opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    codes={}
    counts={}
    for input_batch, cate_batch in tqdm(dataloader):
        with torch.no_grad():
            input_batch = input_batch.cuda()
            for image_idx, input in enumerate(input_batch):
                input_image = input
                cate = cate_batch[image_idx]
                outputs = net.get_test_code(input_image.unsqueeze(0).float())
                # save codes
                if cate not in codes.keys():
                    codes[cate]=outputs['ocodes'][0]
                    counts[cate]=1
                else:
                    codes[cate]+=outputs['ocodes'][0]
                    counts[cate]+=1
    means={}
    for cate in codes.keys():
        means[cate]=codes[cate]/counts[cate]
    torch.save(means,os.path.join(class_embedding_path, 'class_embeddings.pt'))

if __name__ == '__main__':
    run()
