import torch
import os
from argparse import Namespace

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import sys
import json
import lpips
import cv2
import random
import shutil

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.age import AGE



def get_n_distribution(net, transform, means, opts):
    samples=os.listdir(opts.data_path)
    xs0=[]
    xs1=[]
    for s in tqdm(samples):
        cate=s.split('_')[0]
        av_codes=means[cate].cuda()
        from_im = Image.open(os.path.join(opts.data_path,s))
        from_im = from_im.convert('RGB')
        from_im = transform(from_im)
        with torch.no_grad():
            x=net.get_code(from_im.unsqueeze(0).to("cuda").float(), av_codes.unsqueeze(0))
            xs=torch.linalg.lstsq(x['A'], x['odw'][0][:6]).solution
            xs0.append(x[0].squeeze(0).squeeze(0))
            xs1.append(x[1].squeeze(0).squeeze(0))
    codes0=torch.stack(xs0)
    mean_abs0=np.mean(codes0.abs().cpu().numpy(),axis=0)
    mean0=np.mean(codes0.cpu().numpy(),axis=0)
    cov_codes0=codes0.cpu().numpy()
    cov0=np.cov(cov_codes0.T)
    codes1=torch.stack(xs1)
    mean_abs1=np.mean(codes1.abs().cpu().numpy(),axis=0)
    mean1=np.mean(codes1.cpu().numpy(),axis=0)
    cov_codes1=codes1.cpu().numpy()
    cov1=np.cov(cov_codes1.T)
    np.save(opts.n_distribution_path,{'mean0':mean0, 'cov0':cov0, 'mean_abs0':mean_abs0, 'mean1':mean1, 'cov1':cov1, 'mean_abs1':mean_abs1})


def sampler(outputs, opts):
    p=np.load(opts.n_distribution_path,allow_pickle=True).item()
    mean0=p['mean0']
    cov0=p['cov0']
    mean1=p['mean1']
    cov1=p['cov1']
    sampled_x0=np.random.multivariate_normal(mean=mean0, cov=cov0, size=1)
    sampled_x0=torch.from_numpy(sampled_x0).unsqueeze(0).cuda().float()
    sampled_x1=np.random.multivariate_normal(mean=mean1, cov=cov1, size=1)
    sampled_x1=torch.from_numpy(sampled_x1).unsqueeze(0).cuda().float()
    dw=torch.cat((1*torch.matmul(outputs['A'][:3], sampled_x0.transpose(1,2)).squeeze(-1),0.4*torch.matmul(outputs['A'][3:6], sampled_x1.transpose(1,2)).squeeze(-1)), dim=0)
    codes = torch.cat(((dw.unsqueeze(0)+ outputs['ocodes'][:, :6]), outputs['ocodes'][:, 6:]), dim=1)
    return codes



if __name__=='__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)

    #load model
    test_opts = TestOptions().parse()
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
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
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    transform=transforms_dict['transform_inference']


    # get n distribution
    class_embeddings=torch.load(test_opts.class_embedding_path)
    get_n_distribution(net, transform, class_embeddings, test_opts)


    # generate data
    test_data_path=test_opts.test_data_path
    output_path=test_opts.output_path
    os.makedirs(output_path, exist_ok=True)
    for cate in tqdm(os.listdir(test_data_path)):
        os.makedirs(output_path+cate, exist_ok=True)
        from_im = Image.open(test_data_path+cate)
        from_im = from_im.convert('RGB')
        from_im = transform(from_im)
        outputs = net.get_test_code(from_im.unsqueeze(0).to("cuda").float())
        for i in range(50):
            codes=sampler(outputs)
            with torch.no_grad():
                res0 = net.decode(codes, randomize_noise=False, resize=opts.resize_outputs)
            res0 = tensor2im(res0[0])
            im_save_path = os.path.join(output_path, cate, str(i)+'.jpg')
            Image.fromarray(np.array(res0)).save(im_save_path)


    