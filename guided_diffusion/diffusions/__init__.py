from . import base_diffusion
from . import coldmix_diffusion

def get_diffusion(name):
    if name == 'coldmix':
        return coldmix_diffusion.ColdMixDiffusion
    elif name == 'noise_diffusion':
        return coldmix_diffusion.NoiseDiffusion
    elif name == 'base_diffusion':
        return base_diffusion.BaseDiffusion
    else:
        print(f'Warning, unknown diffusion class: {name}. Using BaseDiffusion.')
        return base_diffusion.BaseDiffusion