import torch
import os

import torch.distributed as dist
from torchvision.utils import make_grid
from . import dist_util, logger
from .saving_imgs_utils import save_img,tensor2img
from .train_util import TrainLoop

class Train_noisediff_Loop(TrainLoop):
    def validation_sample_for_start_point(self, data_to_sample, num_samples=8):
        single_sample_loader = torch.utils.data.DataLoader(data_to_sample.dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        single_sample_loader = iter(single_sample_loader)
        # if only_first_batch: # Use only the first batch of the loader every time
        #     sample_condition_data = self.val_samples_data if data_to_sample is self.val_dataset \
        #                             else self.test_samples_data
        #     batch_size = sample_condition_data[0].shape[0]
        #     num_samples = batch_size # sample only batch size samples
        # else:
        #     batch_size = 1
        self.model.eval()

        image_size = self.model.image_size
        # Local setup
        clip_denoised = True

        counter = 0
        mse_total = 0
        # num_sampled_images = 0 # use counter instead
        images_grid, pred_grid = [], []

        while counter < num_samples:
            # sample from the dataloader. may cause different samples each call
            sample_condition_data = next(single_sample_loader)
            batch_size = sample_condition_data[0].shape[0]

            gt_imgs, data_dict = sample_condition_data
            model_kwargs = data_dict
            sample = self.diffusion.p_sample_loop(
                self.model,
                (batch_size, 3, image_size, image_size),
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                noise=data_dict['x_T_end'].to(dtype=torch.float32, device=dist_util.dev()),
                #x_start=gt_imgs.to(dtype=torch.float32, device=dist_util.dev())
            )
            # Copy from image sample code:
            sample = sample.clone()

            # Removed numpy data saving
            #x_start = gt_imgs.to(dtype=torch.float32, device=dist_util.dev())
            #data_dict['x_T_end'] = self.diffusion.q_sample(x_start, torch.tensor([self.diffusion.num_timesteps-1] * batch_size, device=dist_util.dev()))
            res_img = tensor2img(sample)
            save_img(res_img, os.path.join(logger.get_dir(), f"img_sp_pred{counter}_{(self.step + self.resume_step):06d}.png"))

            if self.step == 0:
                save_img(tensor2img(gt_imgs), os.path.join(logger.get_dir(), f"img_sp_x0{counter}.png"))
                save_img(tensor2img(data_dict['x_T_end']), os.path.join(logger.get_dir(), f"img_sp_xT{counter}.png"))
                images_grid.extend([data_dict['x_T_end'], gt_imgs])
            pred_grid.append(sample)
            mse = ((sample - gt_imgs.to(dtype=torch.float32, device=dist_util.dev()))**2/4).mean() # /(2^2) due to dynamic-range -1,1
            mse_total = mse_total+mse
            counter += 1
        mse_total = mse_total/counter

        logger.logkv(f'PSNR_for_start_point', -10 * torch.log10(mse_total))
        pred_grid = make_grid(torch.cat(pred_grid, 0), nrow=1, normalize=False)
        logger.get_logger().logimage(f'img_sp_preds', tensor2img(pred_grid))
        if self.step == 0:
            images_grid = make_grid(torch.cat(images_grid, 0), nrow=2, normalize=False).unsqueeze(0)
            logger.get_logger().logimage(f'img_sp_in_out', images_grid)
        dist.barrier()
        #logger.log("sampling complete")
        self.model.train()