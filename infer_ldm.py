import sys
sys.path.append(".")
sys.path.append('./taming-transformers')

import torch
from omegaconf import OmegaConf
import time

from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

from ldm.models.diffusion.ddim import DDIMSampler

model = get_model()
sampler = DDIMSampler(model)

import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

classes = [25, 187, 448, 992, 108, 282]  # define classes to be sampled here
n_samples_per_class = 1

ddim_steps = 250
ddim_eta = 0.0
scale = 6

use_faster_diffusion = True
if not use_faster_diffusion:  # ldm
    key_time_steps = tuple(range(ddim_steps + 1))
else:  # ldm w/ faster diffusion
    # key_time_steps = (0, 1, 2, 3, 4, 5, 10, 15, 25, 35, 55, 75, 85, 95, 96, 97, 98, 99,
    #                   110, 125, 135, 140, 145, 146, 147, 148, 149, 155, 160, 170, 171, 172, 173, 174, 185, 195, 196, 197, 198, 199,
    #                   210, 222, 223, 224, 235, 247, 248, 249, ddim_steps)
    key_time_steps = (0, 1, 2, 3, 4, 5, 10, 25, 55, 85, 110, 135, 155, 160, 185, 195, 235, 248, 249, ddim_steps)

sampler.model.model.diffusion_model.register_store = {
    'bs': n_samples_per_class,
    'tqdm_disable': True,
    'noise_injection': False if not use_faster_diffusion else True,
    'key_time_steps': key_time_steps,
    'se_step': False,
    'skip_feature': None, 'mid_feature': None,  # store encoder features
    'init_img': None,
    # parallel encoder propagation
    'use_parallel': True,
    'ts_parallel': None, 'steps': [0],  # store time-steps
}

all_samples = list()

x_T = torch.randn((n_samples_per_class, 3, 64, 64), device=model.betas.device)

with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)}
        )

        allt = 0
        for class_label in classes:
            print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor(n_samples_per_class * [class_label])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

            st = time.time()
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples_per_class,
                                             shape=[3, 64, 64],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=x_T)
            allt += (time.time() - st)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            all_samples.append(x_samples_ddim)

        print(f'cost {allt/len(classes)} second / it')

# display as grid
grid = torch.stack(all_samples, 0)
grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=len(classes))

# to image
grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
image = Image.fromarray(grid.astype(np.uint8))

image.save(f"samples.jpg")