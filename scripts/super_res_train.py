"""
Train a super-resolution model.
"""

import argparse

import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.script_util import parse_yaml
from guided_diffusion.script_util import load_folder_path_parse


def main():
    args = create_argparser().parse_args()
    args = parse_yaml(args)
    args.large_size = args.image_size
    if args.load and not args.resume_checkpoint:
        load_folder_path_parse(args)
        args.resume_checkpoint = args.model_path

    dist_util.setup_dist()
    logger.configure(args=args)
    logger.log(f'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_superres_data(
        args.data_dir,
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
        clip_file_path=args.clip_file_path,
    )
    val_data = load_superres_data(
        args.data_dir,
        8, # args.batch_size, todo fix it
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
        deterministic=True,
        clip_file_path=args.clip_file_path,
    )
    test_data = load_superres_data(
        data_dir=args.data_dir_test,
        batch_size=8,  # args.batch_size, todo fix it
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
        deterministic=True,
        clip_file_path=args.clip_file_path_test,
    )
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size, # bug fix
        image_size=args.image_size,
        class_cond=args.class_cond,
        clip_file_path=args.clip_file_path,
    )
    val_data = load_data(
        data_dir=args.data_dir,
        batch_size=args.val_batch_size, # args.batch_size, todo fix it
        image_size=args.large_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        clip_file_path=args.clip_file_path,
    )
    test_data = load_data(
        data_dir=args.data_dir_test,
        batch_size=args.val_batch_size, # args.batch_size, todo fix it
        image_size=args.large_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        clip_file_path=args.clip_file_path_test,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        val_datasets=(val_data, test_data),
        args=args,
    ).run_loop()


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False, deterministic=False, clip_file_path=None):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        deterministic=deterministic,
        clip_file_path=clip_file_path,
    )
    for large_batch, model_kwargs in data:
        #model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        SR_mode=True,
        resume_ema_opt=True,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
