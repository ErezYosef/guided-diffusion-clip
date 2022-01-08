"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.script_util import parse_yaml
from guided_diffusion.image_datasets import load_data
from guided_diffusion.saving_imgs_utils import save_img,tensor2img
from guided_diffusion.script_util import load_folder_path_parse
from guided_diffusion.sample_util import *
def main():
    args = create_argparser().parse_args()
    print(args.config_file)
    #args.config_file = 'sample_config.yaml'
    args = parse_yaml(args)
    load_folder_path_parse(args)
    args.large_size = args.image_size

    dist_util.setup_dist()
    logger.configure(args=args)
    logger.log(f'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    #data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)
    data = load_data(
        data_dir=args.data_dir_test,
        batch_size=args.batch_size,  # args.batch_size
        image_size=args.large_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        clip_file_path=args.clip_file_path_test,
    )
    denoise_start_point = range(500, 1000, 199)
    logger.log("creating samples...")
    all_images = []

    counter=0
    while len(all_images) * args.batch_size < args.num_samples:
        imgs, kwargs = next(data)
        kwargs = process1(kwargs)
        imgs_start = kwargs['img2'].to(dist_util.dev())
        for st in denoise_start_point:
            model_kwargs = kwargs

            denoise_start_point_if = (st, imgs_start)
            #model_kwargs = process1(model_kwargs)
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            sample = diffusion.p_sample_loop(
                model,
                (args.batch_size, 3, args.large_size, args.large_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                denoise_start_point=denoise_start_point_if,
            )
            sample_cp = sample.clone()
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(all_samples, sample)  # gather not supported with NCCL
            for sample in all_samples:
                all_images.append(sample.cpu().numpy())
            logger.log(f"created {len(all_images) * args.batch_size} samples")

            res_img = tensor2img(sample_cp)
            save_img(res_img, os.path.join(logger.get_dir(), f"samples_depth{st}_test{counter}.png"))
            counter+=1
    '''
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
    '''
    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
        config_file='sample_config.yaml',
        denoise_start_point=None,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
