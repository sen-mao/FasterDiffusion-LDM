import argparse
import os
import time
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append('./taming-transformers')
# from taming.models import vqgan

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
)
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    return model

def get_model(config_path, model_path):
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, model_path)
    return model

def main():
    args = create_argparser().parse_args()
    print(args)

    dist_util.setup_dist()
    output_dir = 'output_ldm_eval' if not args.use_faster_diffusion else 'output_ldm_fdiffusion_eval'
    logger.configure(dir=output_dir)

    # Now, download the checkpoint (~1.7 GB). This will usually take 1-2 minutes.
    model = get_model(args.config_path, args.model_path)
    sampler = DDIMSampler(model)
    if not args.use_faster_diffusion:  # ldm
        key_time_steps = tuple(range(args.ddim_steps + 1))
    else:  # ldm w/ faster diffusion
        key_time_steps = (0, 1, 2, 3, 4, 5, 10, 15, 25, 35, 55, 75, 85, 95, 96, 97, 98, 99, 110, 125, 135, 140, 145, 146, 147, 148, 149, 155, 160, 170, 171, 172, 173, 174, 185, 195, 196, 197, 198, 199, 210, 222, 223, 224, 235, 247, 248, 249, args.ddim_steps)
    sampler.model.model.diffusion_model.register_store = {
        'bs': args.batch_size,
        'tqdm_disable': args.tqdm_disable,
        'noise_injection': False if not args.use_faster_diffusion else True,
        'key_time_steps': key_time_steps,
        'se_step': False,
        'skip_feature': None, 'mid_feature': None, # store encoder features
        'init_img': None,
        # parallel encoder propagation
        'use_parallel': True,
        'ts_parallel': None, 'steps': [0],  # store time-steps
    }

    model.to(dist_util.dev())
    # if args.use_fp16:
    #     model.convert_to_fp16()
    model.eval()

    uc = model.get_learned_conditioning(
        {model.cond_stage_key: torch.tensor(args.batch_size * [1000]).to(model.device)}
    )

    logger.log("sampling...")
    all_images = []
    all_labels = []
    start_time = time.time()
    bar_length = 100
    while len(all_images) * args.batch_size < args.num_samples:
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )

        # xc = torch.tensor(args.batch_size * classes)
        c = model.get_learned_conditioning({model.cond_stage_key: classes})

        samples_ddim, _ = sampler.sample(S=args.ddim_steps,
                                         conditioning=c,
                                         batch_size=args.batch_size,
                                         shape=[3, 64, 64],
                                         verbose=False,
                                         unconditional_guidance_scale=args.classifier_scale,
                                         unconditional_conditioning=uc,
                                         eta=args.ddim_eta)

        x_samples_ddim = model.decode_first_stage(samples_ddim)


        sample = ((x_samples_ddim + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)  # (bs, 3, 256, 256)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        # logger.log(f"created {len(all_images) * args.batch_size} samples")

        if dist.get_rank() == 0:
            all_images_num = len(all_images) * args.batch_size
            # calculate the progress bar length
            filled_length = int(bar_length * all_images_num / args.num_samples)
            # calculate the remaining progress bar length
            remaining_length = bar_length - filled_length
            # calculate the elapsed time
            elapsed_time = time.time() - start_time
            if all_images_num > 0:
                remaining_time = elapsed_time * (args.num_samples / all_images_num -1)
            else:
                remaining_time = 0
            print('\rProgress: [{0}{1}] {2:.3f}% Elapsed Time: {3:.2f}s, Remaining Time: {4:.2f}min'.format(
                '=' * filled_length,
                ' ' * remaining_length,
                all_images_num / args.num_samples * 100,
                elapsed_time,
                remaining_time / 60
            ), end='', flush=True)

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        num_samples=50000,
        batch_size=16,
        config_path="../configs/latent-diffusion/cin256-v2.yaml",
        model_path="../models/ldm/cin256-v2/model.ckpt",
        classifier_scale=1.5,
        ddim_eta=0.0,
        ddim_steps=250,
        tqdm_disable=False,
        use_faster_diffusion=False,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()