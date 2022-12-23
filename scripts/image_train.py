"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    create_model_and_diffusion, create_model_new,
    args_to_dict,
    add_dict_to_argparser,
    all_args_to_dict,
)
from guided_diffusion.defaults_and_args import model_and_diffusion_defaults
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.script_util import parse_yaml
from guided_diffusion.base_diffusion import BaseDiffusion
from guided_diffusion.coldmix_diffusion import ColdMixDiffusion
from guided_diffusion.respace_diffusion import SpacedDiffusion
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule

def main():
    args = create_argparser().parse_args()
    args = parse_yaml(args)

    dist_util.setup_dist()
    logger.configure(args=args)

    logger.log(f'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))
    logger.log("creating model and diffusion...")
    #model, diffusion = create_model_and_diffusion(
    #    **args_to_dict(args, model_and_diffusion_defaults().keys()))
    print('pass1')
    model = create_model_new(**all_args_to_dict(args))
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    diffusion_args = all_args_to_dict(args)
    diffusion_args['betas'] = betas
    diffusion = SpacedDiffusion(ColdMixDiffusion, **diffusion_args) # ColdMix or BaseDiffusion

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"creating data loader... dir: {args.data_dir}")
    train_data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    val_data = load_data(
        data_dir=args.data_dir,
        batch_size=8, # args.batch_size, todo fix it
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )
    test_data = load_data(
        data_dir=args.data_dir_test,
        batch_size=8, # args.batch_size, todo fix it
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_data=train_data,
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
        val_dataset=val_data,
        test_dataset=test_data,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        config_file='image_train_config.yaml',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
