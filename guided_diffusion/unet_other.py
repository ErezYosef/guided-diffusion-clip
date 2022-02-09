import torch.nn as nn
from .nn import linear
from guided_diffusion.unet import UNetModel, TimestepEmbedSequential, conv_nd
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

class SRImageModel_Feat(UNetModel):
    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels*2, *args, **kwargs)
        if self.num_classes is not None:
            self.label_emb = nn.Sequential(
            linear(self.num_classes, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.bias_feat = nn.Parameter(th.randn(self.num_classes))
        print('layer_changed !', self.num_classes, self.time_embed_dim)

    def forward(self, x, timesteps, clip_feat=None, clip_feat2=None, img2=None, **kwargs):
        '''
        Xt: noisy img1 at timstep t
        img2: reference/input image
        clip_feat1,2: clip features
        We want to predict X0 from Xt using the clip_feat and img2
        '''
        y = clip_feat.squeeze().float()  # Nx512
        clip_feat2 = clip_feat2.squeeze().float()  # Nx512
        #print('yshape', clip_feat2.shape, self.bias_feat)
        y = y - clip_feat2 + self.bias_feat
        #y = clip_feat2 + self.bias_feat
        #print('yshape', y.shape)
        # print('generating y', y.shape)
        B, C, new_height, new_width = x.shape
        #upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        #spat_feat_repaeted = clip_feat.float().view(B, 512).repeat(1, 32)  # hard coded
        #spat_feat_repaeted_sized = spat_feat_repaeted.view(B, 1, 128,128)

        #x = th.cat([x, spat_feat_repaeted_sized, th.transpose(spat_feat_repaeted_sized, 2, 3)], dim=1)
        x = th.cat([x, img2], dim=1)
        #print('xshape', x.shape)
        return super().forward(x, timesteps, y=y, **kwargs)

class SRImageModel_Feat_cont(UNetModel):
    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels, *args, **kwargs)
        if self.num_classes is not None:
            self.label_emb = nn.Sequential(
            linear(self.num_classes, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )
        print('sizes:', self.num_classes, self.time_embed_dim)

        def init_weights_zero(m):
            if isinstance(m, nn.Linear):
                th.nn.init.zeros_(m.weight)
                th.nn.init.zeros_(m.bias)
        self.bias_feat = nn.Parameter(th.randn(self.num_classes))
        self.label_emb.apply(init_weights_zero)
        #self.cont_setup_after_load_weights_on_first_run = True
        print('layer_changed !', self.num_classes, self.time_embed_dim)

    def forward(self, x, timesteps, clip_feat=None, clip_feat2=None, img2=None, **kwargs):
        y = clip_feat.squeeze().float()  # Nx512
        clip_feat2 = clip_feat2.squeeze().float()  # Nx512
        #print('yshape', clip_feat2.shape, self.bias_feat)
        y = y - clip_feat2 + self.bias_feat
        #print('yshape', y.shape)
        # print('generating y', y.shape)
        B, C, new_height, new_width = x.shape
        #upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        #spat_feat_repaeted = clip_feat.float().view(B, 512).repeat(1, 32)  # hard coded
        #spat_feat_repaeted_sized = spat_feat_repaeted.view(B, 1, 128,128)

        #x = th.cat([x, spat_feat_repaeted_sized, th.transpose(spat_feat_repaeted_sized, 2, 3)], dim=1)
        x = th.cat([x, img2], dim=1)
        #print('xshape', x.shape)
        return super().forward(x, timesteps, y=y, **kwargs)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        super().load_state_dict(state_dict, False)
        self.cont_after_load_weights()

    def cont_after_load_weights(self):
        ch = 64# int(self.channel_mult[0] * self.model_channels)
        in_channels = self.in_channels
        weight = self.input_blocks[0][0].weight.detach().clone()
        bias = self.input_blocks[0][0].bias.detach().clone()
        device = self.input_blocks[0][0].bias.device
        print(weight.shape, bias.shape, device)
        self.input_blocks[0] = TimestepEmbedSequential(conv_nd(2, in_channels*2, ch, 3, padding=1)).to(device)
        th.nn.init.zeros_(self.input_blocks[0][0].weight)
        #th.nn.init.zeros_(self.input_blocks[0][0].bias)
        #self.input_blocks[0][0].weight.fill_(0)
        #self.input_blocks[0][0].bias.fill_(0)
        with th.no_grad():
            self.input_blocks[0][0].weight[:, :in_channels] = weight
            self.input_blocks[0][0].bias[:] = bias
        print(self.input_blocks[0][0].weight.shape)
        print(self.input_blocks[0][0].bias.shape)

