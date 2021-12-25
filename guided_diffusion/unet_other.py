import torch.nn as nn
from .nn import linear
from guided_diffusion.unet import UNetModel
import torch as th

class SpatFeatureModel(UNetModel):
    """
    A UNetModel that performs feature concat to the input.

    Expects an extra kwarg `clip_feat` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels + 2, *args, **kwargs)

    def forward(self, x, timesteps, clip_feat=None, **kwargs):
        B, C, new_height, new_width = x.shape
        #upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        spat_feat_repaeted = clip_feat.float().view(B, 512).repeat(1, 32)  # hard coded
        spat_feat_repaeted_sized = spat_feat_repaeted.view(B, 1, 128,128)

        x = th.cat([x, spat_feat_repaeted_sized, th.transpose(spat_feat_repaeted_sized, 2, 3)], dim=1)
        return super().forward(x, timesteps, **kwargs)

class UNetModel_clip_feat(UNetModel):
    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels, *args, **kwargs)
        if self.num_classes is not None:
            self.label_emb = nn.Sequential(
            linear(self.num_classes, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )
        print('layer_changed !', self.num_classes, self.time_embed_dim)

    def forward(self, x, timesteps, clip_feat=None, **kwargs):
        # forward(self, x, timesteps, y=None, **kwargs)

        y = clip_feat.squeeze().float() # Nx512
        #print('generating y', y.shape)
        return super().forward(x, timesteps, y=y, **kwargs)
