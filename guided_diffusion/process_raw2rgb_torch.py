import torch


def random_gains(deterministic=False):
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    rgb_gain = 1.0 / torch.normal(mean=torch.tensor(0.8), std=torch.tensor(0.1))
    if deterministic:
        t = torch.ones(1)
        return t*0.8, t*2.15, t*1.7

    # Red and blue gains represent white balance.
    red_gain = torch.ones(1).uniform_(1.9, 2.4)
    blue_gain = torch.ones(1).uniform_(1.5, 1.9)
    return rgb_gain, red_gain, blue_gain

def apply_gains(bayer_images, red_gains, blue_gains):
    """Applies white balance gains to a batch of Bayer images."""
    assert bayer_images.shape[1] == 4, "Invalid shape for bayer_images"

    green_gains = torch.ones_like(red_gains)
    gains = torch.stack([red_gains, green_gains, green_gains, blue_gains], dim=0)
    gains = gains.view(1,4,1,1).to(bayer_images.device)
    #print(gains.shape)
    return bayer_images * gains

import torch

def random_ccm(deterministic=False):
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = torch.tensor([[[1.0234, -0.2969, -0.2266],
                              [-0.5625, 1.6328, -0.0469],
                              [-0.0703, 0.2188, 0.6406]],
                             [[0.4913, -0.0541, -0.0202],
                              [-0.613, 1.3513, 0.2906],
                              [-0.1564, 0.2151, 0.7183]],
                             [[0.838, -0.263, -0.0639],
                              [-0.2887, 1.0725, 0.2496],
                              [-0.0627, 0.1427, 0.5438]],
                             [[0.6596, -0.2079, -0.0562],
                              [-0.4782, 1.3016, 0.1933],
                              [-0.097, 0.1581, 0.5181]]])
    num_ccms = len(xyz2cams)
    weights = torch.rand((num_ccms, 1, 1)) * (1e8 - 1e-8) + 1e-8
    if deterministic:
        weights = torch.ones((num_ccms, 1, 1)) * (1e8 - 1e-8) + 1e-8
    weights_sum = torch.sum(weights, dim=0)
    xyz2cam = torch.sum(xyz2cams * weights, dim=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float32)
    rgb2cam = torch.matmul(xyz2cam, rgb2xyz)

    # Normalizes each row.
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    cam2rgb = torch.linalg.inv(rgb2cam)
    return rgb2cam, cam2rgb


def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    assert images.ndim == 4, "Invalid rank for images"
    out = torch.einsum('bchw,mc->bmhw', images, ccms.to(images.device)) # Erez.
    #shape = tf.shape(image)
    #image = tf.reshape(image, [-1, 3])
    #image = tf.tensordot(image, ccm, axes=[[-1], [-1]])
    return out
    images = images.unsqueeze(3)
    ccms = ccms.unsqueeze(1).unsqueeze(1)
    return torch.sum(images * ccms, dim=-1)



def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return torch.clamp(images, min=1e-8) ** (1.0 / gamma)

class Layout(): # enum.Enum
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)

class Debayer3x3(torch.nn.Module):
    """Demosaicing of Bayer images using 3x3 convolutions.

    Compared to Debayer2x2 this method does not use upsampling.
    Instead, we identify five 3x3 interpolation kernels that
    are sufficient to reconstruct every color channel at every
    pixel location.

    We convolve the image with these 5 kernels using stride=1
    and a one pixel reflection padding. Finally, we gather
    the correct channel values for each pixel location. Todo so,
    we recognize that the Bayer pattern repeats horizontally and
    vertically every 2 pixels. Therefore, we define the correct
    index lookups for a 2x2 grid cell and then repeat to image
    dimensions.
    """
    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer3x3, self).__init__()
        self.layout = layout
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    [0, 0.25, 0],
                    [0.25, 0, 0.25],
                    [0, 0.25, 0],

                    [0.25, 0, 0.25],
                    [0, 0, 0],
                    [0.25, 0, 0.25],

                    [0, 0, 0],
                    [0.5, 0, 0.5],
                    [0, 0, 0],

                    [0, 0.5, 0],
                    [0, 0, 0],
                    [0, 0.5, 0],
                ]
            ).view(4, 1, 3, 3),
            requires_grad=False,
        )
        # fmt: on

        self.index = torch.nn.Parameter(
            self._index_from_layout(layout),
            requires_grad=False,
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        B, C, H, W = x.shape

        xpad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
        c = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
        c = torch.cat((c, x), 1)  # Concat with input to give identity kernel Bx5xHxW

        rgb = torch.gather(
            c,
            1,
            self.index.repeat(
                1,
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # expand in batch is faster than repeat
        )
        return rgb

    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
        """Returns a 1x3x2x2 index tensor for each color RGB in a 2x2 bayer tile.

        Note, the index corresponding to the identity kernel is 4, which will be
        correct after concatenating the convolved output with the input image.
        """
        #       ...
        # ... b g b g ...
        # ... g R G r ...
        # ... b G B g ...
        # ... g r g r ...
        #       ...
        # fmt: off
        rggb = torch.tensor(
            [
                # dest channel r
                [4, 2],  # pixel is R,G1
                [3, 1],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [1, 3],  # pixel is R,G1
                [2, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout)


def warp_channels_to_bayer(channels):
    """Warps 4 channels of R, GR, GB, B to Bayer pattern RGGB image."""
    Batch, C, H, W = channels.shape

    # Extract channels
    R = channels[:, 0, :, :]
    GR = channels[:, 1, :, :]
    GB = channels[:, 2, :, :]
    B = channels[:, 3, :, :]

    # Create empty Bayer image
    bayer_image = torch.zeros((Batch, 1, 2 * H, 2 * W), dtype=channels.dtype, device=channels.device)

    # Fill the Bayer image with channels
    bayer_image[:, 0, 0::2, 0::2] = R
    bayer_image[:, 0, 0::2, 1::2] = GR
    bayer_image[:, 0, 1::2, 0::2] = GB
    bayer_image[:, 0, 1::2, 1::2] = B

    return bayer_image

def process(bayer_images, red_gains=None, blue_gains=None, cam2rgbs=None, deterministic=True, min_max=(0,1)):
    bayer_images = (bayer_images - min_max[0]) / (min_max[1] - min_max[0]) # to range [0,1]

    if red_gains is None or blue_gains is None:
        rgb_gain, red_gains, blue_gains = random_gains(deterministic)
    if cam2rgbs is None:
        rgb2cam, cam2rgbs = random_ccm(deterministic)
        #print(rgb2cam.shape, cam2rgbs.shape)
    """Processes a batch of Bayer RGGB images into sRGB images."""
    assert bayer_images.shape[1] == 4, f"Invalid shape for bayer_images, {bayer_images.shape}"
    demosaic = Debayer3x3().to(device=bayer_images.device)

    # White balance.
    bayer_images = apply_gains(bayer_images, red_gains, blue_gains)
    # Demosaic.
    bayer_images = torch.clamp(bayer_images, 0.0, 1.0)
    bayer_images = warp_channels_to_bayer(bayer_images)
    images = demosaic(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, 0.0, 1.0)
    images = gamma_compression(images)

    return images

