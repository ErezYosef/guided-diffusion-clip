import numpy as np
import torch
from .respace import space_timesteps
# from .base_diffusion import b


def SpacedDiffusion(diffusion_class, **kwargs):
    '''
    return a warpper class for diffusion_class. use:
    diffusion = SpacedDiffusion(original_parameters_of_parent_class)
    '''
    class SpacedDiffusion_wrapper(diffusion_class):
        def __init__(self, timestep_respacing, **kwargs):
            original_num_steps = len(kwargs["betas"])
            if not timestep_respacing:
                timestep_respacing = [original_num_steps]
            use_timesteps = space_timesteps(original_num_steps, timestep_respacing)
            self.use_timesteps = set(use_timesteps)
            self.timestep_map = []
            self.original_num_steps = len(kwargs["betas"])

            base_diffusion = diffusion_class(**kwargs)  # pylint: disable=missing-kwoa
            last_alpha_cumprod = 1.0
            new_betas = []
            for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
                if i in self.use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)
            kwargs["betas"] = np.array(new_betas)
            super().__init__(**kwargs)

        def _scale_timesteps(self, ts):
            map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
            new_ts = map_tensor[ts]
            if self.rescale_timesteps:
                new_ts = new_ts.float() * (self.timesteps_scaling_factor / self.original_num_steps)
            return new_ts

    return SpacedDiffusion_wrapper(**kwargs)