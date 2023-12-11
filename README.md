# HiNeRV

This is the repository of the paper "HiNeRV: Video Compression with Hierarchical Encoding-based Neural Representation"  (NeurIPS 2023).

By Ho Man Kwan, Ge Gao, Fan Zhang, Andrew Gower and David Bull

[Project page](https://hmkx.github.io/hinerv/)

[arXiv](https://arxiv.org/abs/2306.09818)

## Dependencies
```
accelerate==0.23.0
deepspeed==0.11.1
pytorch-msssim==1.0.0
timm==0.9.7
torch==2.1.0
torchac==0.9.3
torchvision==0.16.0
```


## Examples
### Prepare the dataset
This implementation requires first converting the videos into PNGs. For example, you can use [FFMpeg](https://www.ffmpeg.org/):
```
mkdir video
ffmpeg -video_size 1920x1080 -pixel_format yuv420p -i video.yuv video/%4d.png
```

The UVG dataset can be downloaded [here](https://ultravideo.fi/dataset.html).

### Training
To train HiNeRV-S with the UVG dataset (ReadySetGo sequence) and the video compression setting:
```
dataset_dir=~/Datasets/UVG/1920x1080
dataset_name=ReadySetGo
output=~/Models/HiNeRV
train_cfg=$(cat "cfgs/train/hinerv_1920x1080.txt")
model_cfg=$(cat "cfgs/models/uvg-hinerv-s_1920x1080.txt")
accelerate launch --mixed_precision=fp16 --dynamo_backend=inductor hinerv_main.py \
  --dataset ${dataset_dir} --dataset-name ${dataset_name} --output ${output} \
  ${train_cfg} ${model_cfg} --batch-size 144 --eval-batch-size 1 --grad-accum 1 --log-eval false --seed 0
```

The output will be saved into a new folder in the output path, e.g. ~/Models/HiNeRV/HiNeRV-20231030-032238-133f0dfc. This path can be used for resuming training/loading bitstream directly.

Please note that the batch size refers to the number of patches. Make sure to adjust it accordingly if you have changed the patch size (see 'cfgs/train/hinerv_1920x1080.txt' for details). While the original configuration uses 120x120 patches, using larger patches (e.g., 'cfgs/train/hinerv_1920x1080_480x360.txt' uses 480x360 patches) reduces overhead but may slightly impact performance.


To save the model outputs into images, set --log-eval to true.


### Evaluation
To evaluate with the compressed bitstream:
```
dataset_dir=~/Datasets/UVG/1920x1080
dataset_name=ReadySetGo
output=~/Models/HiNeRV
train_cfg=$(cat "cfgs/train/hinerv_1920x1080.txt")
model_cfg=$(cat "cfgs/models/uvg-hinerv-s_1920x1080.txt")
checkpoint_path=~/Models/HiNeRV/HiNeRV-20231030-032238-133f0dfc
accelerate launch --mixed_precision=fp16 --dynamo_backend=inductor hinerv_main.py \
  --dataset ${dataset_dir} --dataset-name ${dataset_name} --output ${output} \
  ${train_cfg} ${model_cfg} --batch-size 144 --eval-batch-size 1 --grad-accum 1 --log-eval false --seed 0 \
  --bitstream ${checkpoint_path} --bitstream-q 6 --eval-only
```

checkpoint_path is the path of the trained model directory, and bitstream-q is the quantization level.


### Other settings
To train HiNeRV-S with the 37 epochs setting (no pruning/quantization):
```
dataset_dir=~/Datasets/UVG/1920x1080
dataset_name=ReadySetGo
output=~/Models/HiNeRV
train_cfg=$(cat "cfgs/train/hinerv_1920x1080_37e_no-compress.txt")
model_cfg=$(cat "cfgs/models/uvg-hinerv-s_1920x1080.txt")
accelerate launch --mixed_precision=fp16 --dynamo_backend=inductor hinerv_main.py \
  --dataset ${dataset_dir} --dataset-name ${dataset_name} --output ${output} \
  ${train_cfg} ${model_cfg} --batch-size 144 --eval-batch-size 1 --grad-accum 1 --log-eval false --seed 0
```


# Results
This implementation has slightly improved average performance compared to the original one. The results for both the original and this version will be provided in the 'results' folder.


## Acknowledgement
Part of the implementation is based on the code from [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) and [HNeRV](https://github.com/haochen-rye/HNeRV).


## Citation
Please consider citing our work if you find that it is useful.
```
@misc{kwan2023hinerv,
    title={HiNeRV: Video Compression with Hierarchical Encoding based Neural Representation}, 
    author={Ho Man Kwan and Ge Gao and Fan Zhang and Andrew Gower and David Bull},
    year={2023},
    eprint={2306.09818},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```
