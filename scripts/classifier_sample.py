"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.script_util import parse_yaml
from guided_diffusion.image_datasets import load_data
from guided_diffusion.saving_imgs_utils import save_img,tensor2img
from guided_diffusion.script_util import load_folder_path_parse
from guided_diffusion.sample_util import *
UNSUP = True
if UNSUP:
    from guided_diffusion.img_txt_dataset import load_data
# CUDA_VISIBLE_DEVICES=0 python scripts/classifier_samole.py -f diff_est -d tstrun_factor0
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
)
import torchvision
import clip

def main():
    args = create_argparser().parse_args()
    print(args.config_file)
    args = parse_yaml(args)
    load_folder_path_parse(args)
    args.large_size = args.image_size
    args.main_path = os.path.join(args.main_path, args.sub_dir_tstsave)

    dist_util.setup_dist()
    logger.configure(args=args)
    logger.log(f'\n\t'.join(f'{k} = {v}' for k, v in vars(args).items()))


    logger.log("creating model and diffusion...")
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

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.large_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        clip_file_path=args.clip_file_path,
    )


    logger.log("loading classifier...")
    '''
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(dist_util.load_state_dict(args.classifier_path, map_location="cpu"))
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
       classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
    '''
    resize = torchvision.transforms.Resize(224, interpolation=3)  # 3 means PIL.Image.BICUBIC
    mean = [2 * x - 1 for x in (0.48145466, 0.4578275, 0.40821073)]
    std = [2 * x for x in (0.26862954, 0.26130258, 0.27577711)]
    normlize = torchvision.transforms.Normalize(mean, std)
    cosine_sim = torch.nn.CosineSimilarity()
    clip_model, preprocess = clip.load("ViT-B/32", device=dist_util.dev())
    clip_model.eval()
    @torch.enable_grad()
    def cond_fn(x, t, **model_kwargs):
        x = x.detach().requires_grad_()
        unscaled_timestep = (t * (diffusion.num_timesteps / 1000)).long()
        t = unscaled_timestep

        out = diffusion.p_mean_variance(model, x, t, clip_denoised=False, model_kwargs=model_kwargs)
        #out = model_kwargs['p_mean_var']

        #fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
        x_in = out["pred_xstart"] #* fac + x * (1 - fac)
        # x_in = out["pred_xstart"]
        #x_in = x_in.detach().requires_grad_()

        #loss = torch.tensor(0)
        if args.clip_guidance_lambda != 0:
            clip_in = normlize(resize(x_in+model_kwargs['img']))
            image_embeds = clip_model.encode_image(clip_in).float()
            d_img = image_embeds - model_kwargs['clip_feat_img_save']
            d_txt = model_kwargs['clip_feat']-model_kwargs['clip_feat2']
            loss = 1 - cosine_sim(d_img, d_txt).mean() * args.clip_guidance_lambda
            #clip_loss = (x_in, text_embed) * self.args.clip_guidance_lambda
            #loss = loss + clip_loss
            #self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())
        if args.l1_lambda != 0:
            loss = loss + torch.abs(x_in).mean() * args.l1_lambda

        return -torch.autograd.grad(loss, x)[0]


    logger.log("sampling...")
    all_images = []
    #all_labels = []
    counter=0
    while len(all_images) * args.batch_size < args.num_samples:
        imgs, kwargs = next(data)
        model_kwargs = kwargs

        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        model_kwargs['loggerdir'] = logger.get_dir()
        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample_cp = sample.clone()
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        add_diff_tosave = True
        sample_cp = sample_cp.cpu()
        # sample_cp_nobias = sample_cp - torch.mean(sample_cp, (2,3), keepdim=True)
        if add_diff_tosave:
            imgs_target = model_kwargs['img'].cpu()
            sample_cp = sample_cp + model_kwargs['img2'].cpu()
        res_img = tensor2img(sample_cp)
        save_img(res_img, os.path.join(logger.get_dir(), f"samples_test{counter}.png"))
        save_img(tensor2img(imgs_target), os.path.join(logger.get_dir(), f"img{counter}_tinput0.png"))
        counter += 1
        #all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")
    '''
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)
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
        classifier_path="",
        classifier_scale=1.0,
        config_file='classifier_sample_config.yaml',
        denoise_start_point=None,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
