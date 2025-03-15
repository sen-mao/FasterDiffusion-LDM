# Official Implementations "Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models" for LDM (NeurIPS'24)
<a href='https://arxiv.org/abs/2312.09608'><img src='https://img.shields.io/badge/ArXiv-2306.05414-red'></a>

[//]: # (> **Faster Diffusion: Rethinking the Role of the Encoder for Diffusion Model Inference**)

[//]: # (>)

[//]: # (> [Senmao Li]&#40;https://github.com/sen-mao&#41;, [Taihang Hu]&#40;https://github.com/hutaiHang&#41;, [Joost van de Weijer]&#40;https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl=en&oi=sra&#41;, [Fahad Shahbaz Khan]&#40;https://sites.google.com/view/fahadkhans/home&#41;, [Tao Liu]&#40;ltolcy0@gmail.com&#41;, [Linxuan Li]&#40;https://github.com/Potato-lover&#41;, [Shiqi Yang]&#40;https://www.shiqiyang.xyz/&#41;, [Yaxing Wang]&#40;https://yaxingwang.netlify.app/author/yaxing-wang/&#41;, [Ming-Ming Cheng]&#40;https://mmcheng.net/&#41;, [Jian Yang]&#40;https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ&hl=en&#41;)

[//]: # (>)

The official codebase for [FasterDiffusion](https://arxiv.org/abs/2312.09608) accelerates [LDM](https://github.com/CompVis/latent-diffusion/tree/main) with **~2.36x** speedup.

[//]: # (## Introduction)

<img width="800" alt="image" src="assets/infer_ldm.jpg">

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
MODEL_FLAGS="--batch_size 4 --num_samples 50000 --classifier_scale 6 --ddim_eta 0.0 --tqdm_disable True --use_faster_diffusion True"
mpiexec -n $NUM_GPUS python scripts/image_sample.py $MODEL_FLAGS --config_path $CONFIG_PATH --model_path $MODEL_PATH
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz

```

## Performance

| Model                  | Dataset | Resolution  | FID&darr; | sFID&darr; | IS&uarr; | Precision&uarr; | Recall&uarr; | s/image&darr; |
|------------------------|:--------:|:---------:|:---------:|:----------:|:--------:|:---------------:|:------------:|:-------------:|
| LDM                    | ImageNet |  256x256  |   3.60    |     --     |  247.67  |      0.870      |    0.480     |      --       |
| LDM*                   | ImageNet | 256x256 |   3.39    |    5.14    |  204.57  |      0.825      |    0.534     |     7.951     |
| LDM w/ FasterDiffusion | ImageNet | 256x256  |   4.09    |    5.99    |  207.49  |      0.848      |    0.482     |     3.373     |

\* Denotes the reproduced results.

# Visualization

Run the `infer_ldm.py` to generate images with FasterDiffusion.

## BibTeX

```
@article{li2024faster,
  title={Faster diffusion: Rethinking the role of the encoder for diffusion model inference},
  author={Li, Senmao and Hu, Taihang and van de Weijer, Joost and Shahbaz Khan, Fahad and Liu, Tao and Li, Linxuan and Yang, Shiqi and Wang, Yaxing and Cheng, Ming-Ming and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={85203--85240},
  year={2024}
}

@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

This codebase is built based on [LDM](https://github.com/CompVis/latent-diffusion/tree/main), and references both [ADM](https://github.com/openai/guided-diffusion/tree/main) and [MDT](https://github.com/sail-sg/MDT/tree/main) code. Thanks very much.


