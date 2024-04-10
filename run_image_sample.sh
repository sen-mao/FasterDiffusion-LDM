#!/bin/bash

export NCCL_P2P_DISABLE=1

CONFIG_PATH=configs/latent-diffusion/cin256-v2.yaml
MODEL_PATH=models/ldm/cin256-v2/model.ckpt
NUM_GPUS=8

echo 'Class-conditional ldm with sampling for ImageNet256x256:'
export OPENAI_LOGDIR=output_ldm_eval
MODEL_FLAGS="--batch_size 16 --num_samples 50000 --classifier_scale 1.5 --ddim_eta 0.0 --tqdm_disable True --use_faster_diffusion False"
mpiexec -n $NUM_GPUS python scripts/image_sample.py $MODEL_FLAGS --config_path $CONFIG_PATH --model_path $MODEL_PATH
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz

echo 'Class-conditional ldm with faster-diffusion sampling for ImageNet256x256:'
export OPENAI_LOGDIR=output_ldm_fdiffusion_eval
MODEL_FLAGS="--batch_size 4 --num_samples 50000 --classifier_scale 1.5 --ddim_eta 0.0 --tqdm_disable True --use_faster_diffusion True"
mpiexec -n $NUM_GPUS python scripts/image_sample.py $MODEL_FLAGS --config_path $CONFIG_PATH --model_path $MODEL_PATH
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz
