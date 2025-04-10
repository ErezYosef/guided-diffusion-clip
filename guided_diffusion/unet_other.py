import torch.nn
import torch.nn as nn
from .nn import linear
from guided_diffusion.unet import UNetModel, TimestepEmbedSequential, conv_nd
import torch as th
from .unet import ResBlock, normalization, zero_module, convert_module_to_f16, convert_module_to_f32, Downsample, Upsample, timestep_embedding
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


class UNetModelConv(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param ndconv_dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        **kwargs
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels # 3
        self.model_channels = model_channels # 64
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions # 8,16
        self.dropout = dropout
        self.channel_mult = channel_mult # 1.1.2.3.4
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        ndconv_dims = dims
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        curr_in_channels = input_ch = int(channel_mult[0] * model_channels) # 1*64
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(ndconv_dims, in_channels, curr_in_channels, 3, padding=1))])
        self._feature_size = curr_in_channels
        input_block_chans = [curr_in_channels] # save num channels of model
        downscale_factor = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        curr_in_channels,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=ndconv_dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                curr_in_channels = int(mult * model_channels)

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += curr_in_channels
                input_block_chans.append(curr_in_channels)
            if level != len(channel_mult) - 1: # end of resolution, perform downsample
                out_ch = curr_in_channels
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            curr_in_channels,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=ndconv_dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            curr_in_channels, conv_resample, dims=ndconv_dims, out_channels=out_ch
                        )
                    )
                )
                curr_in_channels = out_ch
                input_block_chans.append(curr_in_channels)
                downscale_factor *= 2
                self._feature_size += curr_in_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                curr_in_channels,
                time_embed_dim,
                dropout,
                dims=ndconv_dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ResBlock(
                curr_in_channels,
                time_embed_dim,
                dropout,
                dims=ndconv_dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += curr_in_channels

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        curr_in_channels + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=ndconv_dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                curr_in_channels = int(model_channels * mult)

                if level and i == num_res_blocks: # Do upsample
                    out_ch = curr_in_channels
                    layers.append(
                        ResBlock(
                            curr_in_channels,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=ndconv_dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(curr_in_channels, conv_resample, dims=ndconv_dims, out_channels=out_ch)
                    )
                    downscale_factor //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += curr_in_channels

        self.out = nn.Sequential(
            normalization(curr_in_channels),
            nn.SiLU(),
            zero_module(conv_nd(ndconv_dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (self.num_classes is not None),\
            f"must specify y if and only if the model is class-conditional, {y}== {self.num_classes}"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0], f'{y.shape} != {x.shape}'
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class UNetModelConvClean(UNetModel):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param ndconv_dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        torch.nn.Module.__init__(self)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        #self.image_size = image_size
        self.in_channels = in_channels # 3
        self.model_channels = model_channels # 64
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions # 8,16
        self.dropout = dropout
        self.channel_mult = channel_mult # 1.1.2.3.4
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        ndconv_dims = dims
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        curr_in_channels = input_ch = int(channel_mult[0] * model_channels) # 1*64
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(ndconv_dims, in_channels, curr_in_channels, 3, padding=1))])
        self._feature_size = curr_in_channels
        input_block_chans = [curr_in_channels] # save num channels of model
        downscale_factor = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        curr_in_channels,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=ndconv_dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm, ) ]
                curr_in_channels = int(mult * model_channels)

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += curr_in_channels
                input_block_chans.append(curr_in_channels)
            if level != len(channel_mult) - 1: # end of resolution, perform downsample
                out_ch = curr_in_channels
                if resblock_updown:
                    block = ResBlock(
                                curr_in_channels,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=ndconv_dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,)
                else:
                    block = Downsample(curr_in_channels, conv_resample, dims=ndconv_dims, out_channels=out_ch)
                self.input_blocks.append(TimestepEmbedSequential(block))
                curr_in_channels = out_ch
                input_block_chans.append(curr_in_channels)
                downscale_factor *= 2
                self._feature_size += curr_in_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                curr_in_channels,
                time_embed_dim,
                dropout,
                dims=ndconv_dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,),
            ResBlock(
                curr_in_channels,
                time_embed_dim,
                dropout,
                dims=ndconv_dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,), )
        self._feature_size += curr_in_channels

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        curr_in_channels + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=ndconv_dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,)]
                curr_in_channels = int(model_channels * mult)

                if level and i == num_res_blocks: # Do upsample
                    out_ch = curr_in_channels
                    if resblock_updown:
                        block = ResBlock(
                            curr_in_channels,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=ndconv_dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,)
                    else:
                        block = Upsample(curr_in_channels, conv_resample, dims=ndconv_dims, out_channels=out_ch)
                    layers.append(block)
                    downscale_factor //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += curr_in_channels

        self.out = nn.Sequential(
            normalization(curr_in_channels),
            nn.SiLU(),
            zero_module(conv_nd(ndconv_dims, input_ch, out_channels, 3, padding=1)),
        )