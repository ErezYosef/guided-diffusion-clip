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
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.script_util import parse_yaml
from guided_diffusion.image_datasets import load_data
from guided_diffusion.saving_imgs_utils import save_img,tensor2img
from guided_diffusion.script_util import load_folder_path_parse
from guided_diffusion.sample_util import *

# CUDA_VISIBLE_DEVICES=0 python scripts/image_sample.py -f in_adagn -d tstrun_adagn


def main():
    args = create_argparser().parse_args()
    print(args.config_file)
    #args.config_file = 'sample_config.yaml'
    args = parse_yaml(args)
    load_folder_path_parse(args)
    args.main_path = os.path.join(args.main_path, args.sub_dir_tstsave)

    dist_util.setup_dist()
    logger.configure(args=args)
    logger.log(f'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
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
        clip_file_path=args.clip_file_path_test,
    )
    logger.log("sampling...")
    all_images = []
    all_labels = []
    counter = 0
    while len(all_images) * args.batch_size < args.num_samples:
        # model_kwargs = {}
        imgs, kwargs = next(data)
        kwargs = add_delta_imgimg(kwargs)
        model_kwargs = kwargs
        imgs_start = kwargs['img2']
        print(imgs_start.shape,kwargs['clip_feat'].shape )
        print('args.denoise_start_point:', args.denoise_start_point, type(args.denoise_start_point))
        if args.denoise_start_point != -1:
            model_kwargs['img2'] = imgs_start.to(dist_util.dev())
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
            denoise_start_point=args.denoise_start_point
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
        '''
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        '''
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    '''
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
    '''
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
        denoise_start_point=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
