import numpy as np
import math
from torchvision.utils import make_grid
import cv2
from .process_raw2rgb_torch import process

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1), one_line=False):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    # if tensor.shape[1] ==4:
    #     #print('size4 - change to RGB')
    #     tensor = process(tensor)
    #     # tensor = tensor[:, (0,1,3)]
    #     # tensor = (tensor/tensor.max())**0.5
    n_dim = tensor.dim()
    if n_dim == 4:
        if tensor.shape[1] == 4:
            tensor = process(tensor)
        n_img = len(tensor)
        line = int(math.sqrt(n_img)) if not one_line else int(n_img)
        img_np = make_grid(tensor, nrow=line, normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        if tensor.shape[0] == 4:
            tensor = process(tensor.unsqueeze(0)).squeeze(0)
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def save_img(img, img_path, mode='RGB'):
    # save_img(hr_img, f'{result_path}/{current_step}_{idx}_hr.png')
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
