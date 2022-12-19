

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        model_var_type_name='learned_sigma',
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        loss_type_name='mse',
        model_mean_type_name='epsilon',
        rescale_timesteps=False,
    )

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res
