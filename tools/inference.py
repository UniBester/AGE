import torch
import os
from argparse import Namespace

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import sys
import random

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from utils.common import tensor2im
from options.test_options import TestOptions
from models.age import AGE


def get_n_distribution(net, transform, class_embeddings, opts):
    samples=os.listdir(opts.train_data_path)
    xs=[]
    for s in tqdm(samples):
        cate=s.split('_')[0]
        av_codes=class_embeddings[cate].cuda()
        from_im = Image.open(os.path.join(opts.train_data_path,s))
        from_im = from_im.convert('RGB')
        from_im = transform(from_im)
        with torch.no_grad():
            x=net.get_code(from_im.unsqueeze(0).to("cuda").float(), av_codes.unsqueeze(0))['x']
            x=torch.stack(x)
            xs.append(x)
    codes=torch.stack(xs).squeeze(2).squeeze(2).permute(1,0,2).cpu().numpy()
    mean=np.mean(codes,axis=1)
    mean_abs=np.mean(np.abs(codes),axis=1)
    cov=[]
    for i in range(codes.shape[0]):
        cov.append(np.cov(codes[i].T))
    os.makedirs(opts.n_distribution_path, exist_ok=True)
    np.save(os.path.join(opts.n_distribution_path, 'n_distribution.npy'),{'mean':mean, 'mean_abs':mean_abs, 'cov':cov})


def sampler(outputs, dist, opts):
    means=dist['mean']
    means_abs=dist['mean_abs']
    covs=dist['cov']
    one = torch.ones_like(torch.from_numpy(means[0]))
    zero = torch.zeros_like(torch.from_numpy(means[0]))
    dws=[]
    groups=[[0,1,2],[3,4,5]]
    for i in range(means.shape[0]):
        x=torch.from_numpy(np.random.multivariate_normal(mean=means[i], cov=covs[i], size=1)).float().cuda()
        mask = torch.where(torch.from_numpy(means_abs[i])>opts.beta, one, zero).cuda()
        x=x*mask
        for g in groups[i]:
            dw=torch.matmul(outputs['A'][g], x.transpose(0,1)).squeeze(-1)
            dws.append(dw)
    dws=torch.stack(dws)
    codes = torch.cat(((opts.alpha*dws.unsqueeze(0)+ outputs['ocodes'][:, :6]), outputs['ocodes'][:, 6:]), dim=1)
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


    # get n distribution (only needs to be executed once)
    # class_embeddings=torch.load(os.path.join(test_opts.class_embedding_path, 'class_embeddings.pt'))
    # get_n_distribution(net, transform, class_embeddings, test_opts)



    # generate data
    dist=np.load(os.path.join(opts.n_distribution_path, 'n_distribution.npy'), allow_pickle=True).item()
    test_data_path=test_opts.test_data_path
    output_path=test_opts.output_path
    os.makedirs(output_path, exist_ok=True)
    from_ims = os.listdir(test_data_path)
    for from_im_name in from_ims:
        for j in tqdm(range(test_opts.n_images)):
            from_im = Image.open(os.path.join(test_data_path, from_im_name))
            from_im = from_im.convert('RGB')
            from_im = transform(from_im)
            outputs = net.get_test_code(from_im.unsqueeze(0).to("cuda").float())
            codes=sampler(outputs, dist, test_opts)
            with torch.no_grad():
                res0 = net.decode(codes, randomize_noise=False, resize=opts.resize_outputs)
            res0 = tensor2im(res0[0])
            im_save_path = os.path.join(output_path, from_im_name+'_'+str(j)+'.jpg')
            Image.fromarray(np.array(res0)).save(im_save_path)


    