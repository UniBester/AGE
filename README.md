# Attribute Group Editing for Reliable Few-shot Image Generation

## Description   
Implementation of AGE. Our code is modified from pSp.




## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3

### Installation
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/environment.yaml`.


### Pretrained pSp
Here, we use pSp to find the latent code of real images in the latent domain of a pretrained StyleGAN generator. Follow the instructions in the follow link to train a pSp model firsly.
``` 
git clone https://github.com/eladrich/pixel2style2pixel.git
cd pixel2style2pixel
```

## Training
### Preparing your Data
  - There are two types of data structure needed. The first type is that data of different categories are saved in the same folder, which is referred to as `data_without_cate`. The Second type is that data of different categories are saved in different folders, which is referred to as `data_with_cate`.
  - Refer to `configs/paths_config.py` to define the necessary data paths and model paths for training and evaluation. 
  - Refer to `configs/transforms_config.py` for the transforms defined for each dataset/experiment. 
  - Finally, refer to `configs/data_configs.py` for the source/target data paths for the train and test sets
    as well as the transforms.

#### Get Class Embedding
To train AGE, the class embedding of each category in both train and test split should be get first.
```
python scripts/get_class_embedding.py \
--codes_path=/path/to/save/classs/embeddings \
--psp_checkpoint_path=/path/to/pretrained/pSp/checkpoint \
--data_path=/path/to/training/data \
--test_batch_size=4 \
--test_workers=4
```



### Training pSp
The main training script can be found in `scripts/train.py`.   
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.  
Additionally, if you have tensorboard installed, you can visualize tensorboard logs in `opts.exp_dir/logs`.

#### **Training the pSp Encoder**
```
#set GPUs to use.
export CUDA_VISIBLE_DEVICES=0,1,2,3

#begin training.
python scripts/train.py \
--dataset_type=af_encode \
--exp_dir=/path/to/experiment/output \
--workers=8 \
--batch_size=8 \
--valid_batch_size=8 \
--valid_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--start_from_latent_avg \
--l2_lambda=1 \
--sparse_lambda=0.005 \
--orthogonal_lambda=0.0005 \
--A_length=100 \
--psp_checkpoint_path=/path/to/pretrained/pSp/checkpoint \
--class_embedding_path=/path/to/class_embeddings 
```


## Testing
### Inference
Having trained your model, you can use `scripts/generate_samples.py` to apply the model on a set of images.   
For example, 
```
python scripts/generate_samples.py \
--output_path=/path/to/output \
--checkpoint_path=/path/to/checkpoint \
--data_path=/path/to/data_with_cate \
--test_batch_size=4 \
--test_workers=4 \
--codes_path=/path/to/train/average_code
```
