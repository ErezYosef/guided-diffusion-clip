"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    create_model_and_diffusion, create_model_new,
    add_dict_to_argparser,
    args_to_dict,
    all_args_to_dict,
)
from guided_diffusion.defaults_and_args import model_and_diffusion_defaults
from guided_diffusion.script_util import parse_yaml
from guided_diffusion.image_datasets import load_data
from guided_diffusion.saving_imgs_utils import save_img,tensor2img
from guided_diffusion.script_util import load_folder_path_parse

from guided_diffusion.base_diffusion import BaseDiffusion
from guided_diffusion.respace_diffusion import SpacedDiffusion
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule

# CUDA_VISIBLE_DEVICES=0 python scripts/image_sample.py -f in_adagn -d tstrun_adagn


def main():
    args = create_argparser().parse_args()
    print(args.config_file)
    #args.config_file = 'sample_config.yaml'
    args = parse_yaml(args)
    loaded_folder_name = load_folder_path_parse(args)
    # args.main_path = os.path.join(args.main_path, args.sub_dir_tstsave)

    dist_util.setup_dist()
    logger.configure(args=args, loaded_folder_name=loaded_folder_name)
    logger.log(f'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))

    logger.log("creating model and diffusion...")
    # model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model = create_model_new(**all_args_to_dict(args))
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    diffusion_args = all_args_to_dict(args)
    diffusion_args['betas'] = betas
    diffusion = SpacedDiffusion(BaseDiffusion, **diffusion_args)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data = load_data(
        data_dir=args.data_dir_test,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
    )
    logger.log("sampling...")
    all_images = []
    all_labels = []
    counter = 0
    print('args.diffusion_start_point:', args.diffusion_start_point)
    while len(all_images) * args.batch_size < args.num_samples:
        # model_kwargs = {}
        imgs, kwargs = next(data)
        model_kwargs = kwargs
        if args.diffusion_start_point != -1:
            pass # do some
        # model_kwargs = process2(model_kwargs)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            diffusion_start_point=args.diffusion_start_point
        )
        sample_cp = sample.clone()
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        res_img = tensor2img(sample_cp)
        save_img(res_img, os.path.join(logger.get_dir(), f"samples_test{counter}.png"))
        save_target_images = True
        if save_target_images:
            res_img = tensor2img(imgs)
            save_img(res_img, os.path.join(logger.get_dir(), f"target_{counter}.png"))
        counter += 1


        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        logger.log(f"created {len(all_images) * args.batch_size} samples")


    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        config_file='image_sample_config.yaml',
        diffusion_start_point=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
