# Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models

<a href='https://arxiv.org/abs/2312.09608'><img src='https://img.shields.io/badge/ArXiv-2306.05414-red'></a>

> **Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models**
>
> [Senmao Li](https://github.com/sen-mao)\*, [Taihang Hu](https://github.com/hutaiHang)\*, [Fahad Khan](https://sites.google.com/view/fahadkhans/home), [Linxuan Li](https://github.com/Potato-lover), [Shiqi Yang](https://www.shiqiyang.xyz/), [Yaxing Wang](https://yaxingwang.netlify.app/author/yaxing-wang/), [Ming-Ming Cheng](https://mmcheng.net/), [Jian Yang](https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ&hl=en)
>
> ***Denotes equal contribution.**

The official codebase for [FasterDiffusion](https://arxiv.org/abs/2312.09608) accelerates [LDM](https://github.com/CompVis/latent-diffusion/tree/main) with a 6x speedup.

## Requirements

A suitable conda environment named `ldm-faster-diffusion` can be created
and activated with:


```
conda env create -f environment.yaml

conda activate ldm-faster-diffusion
```

Please follow the instructions in the [latent_imagenet_diffusion.ipynb](https://github.com/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb) to download the checkpoint (`models/ldm/cin256-v2/model.ckpt` with ~1.7 GB).

## Sampling and Evaluation ('run_image_sample.sh')

LDM provides a script for sampling from [**class-conditional ImageNet** with Latent Diffusion Models](https://github.com/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb).
The sample code is modified from this code, and the 50k sampling results are saved in the same data format as [ADM](https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py).
The evaluation code is obtained from [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations), and the evaluation environment is already included in the ldm-faster-diffusion environment.

```shell
#!/bin/bash

export NCCL_P2P_DISABLE=1

CONFIG_PATH=configs/latent-diffusion/cin256-v2.yaml
MODEL_PATH=models/ldm/cin256-v2/model.ckpt
NUM_GPUS=8

echo 'Class-conditional ldm sampling for ImageNet256x256:'
export OPENAI_LOGDIR=output_ldm_eval
MODEL_FLAGS="--batch_size 16 --num_samples 50000 --classifier_scale 1.5 --ddim_eta 0.0 --tqdm_disable True --use_faster_diffusion False"
mpiexec -n $NUM_GPUS python scripts/image_sample.py $MODEL_FLAGS --config_path $CONFIG_PATH --model_path $MODEL_PATH
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz

echo 'Class-conditional ldm with faster-diffusion sampling for ImageNet256x256:'
export OPENAI_LOGDIR=output_ldm_fdiffusion_eval
MODEL_FLAGS="--batch_size 4 --num_samples 50000 --classifier_scale 1.5 --ddim_eta 0.0 --tqdm_disable True --use_faster_diffusion True"
mpiexec -n $NUM_GPUS python scripts/image_sample.py $MODEL_FLAGS --config_path $CONFIG_PATH --model_path $MODEL_PATH
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz

```


## BibTeX

```
@article{li2023faster,
  title={Faster diffusion: Rethinking the role of unet encoder in diffusion models},
  author={Li, Senmao and Hu, Taihang and Khan, Fahad Shahbaz and Li, Linxuan and Yang, Shiqi and Wang, Yaxing and Cheng, Ming-Ming and Yang, Jian},
  journal={arXiv preprint arXiv:2312.09608},
  year={2023}
}

@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{https://doi.org/10.48550/arxiv.2204.11824,
  doi = {10.48550/ARXIV.2204.11824},
  url = {https://arxiv.org/abs/2204.11824},
  author = {Blattmann, Andreas and Rombach, Robin and Oktay, Kaan and Ommer, Björn},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Retrieval-Augmented Diffusion Models},
  publisher = {arXiv},
  year = {2022},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Acknowledgement

This codebase is built based on [LDM](https://github.com/CompVis/latent-diffusion/tree/main), and references both [ADM](https://github.com/openai/guided-diffusion/tree/main) and [MDT](https://github.com/sail-sg/MDT/tree/main) code. Thanks very much.


