import torch
import math
import random
import os

from glob import glob
#from natsort import natsorted
import h5py
import scipy.io as sio

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

def warp_DataLoader(loader):
    while True:
        yield from loader

def load_data_sidd(
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    workers=1,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    #all_files = _list_image_files_recursively(data_dir)

    dataset = siddDataset(
        image_size,
        data_dir,
        random_crop=random_crop,
        random_flip=random_flip,
        deterministic=deterministic
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=not deterministic, num_workers=workers, drop_last=True)
    return loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class siddDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        deterministic=False,
    ):
        super().__init__()
        self.resolution = resolution
        #self.local_images = image_paths[shard:][::num_shards]
        self.data_path = image_paths
        self.data_path = '/data2/erez/datasets/denoising/sidd_raw/medium_raw/SIDD_Medium_Raw/Data'
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.deterministic = deterministic

        self.map_images()

    def map_images(self):

        #self.noisy_files = natsorted(glob(os.path.join(self.data_path, '*', '*NOISY*')))
        self.noisy_files = sorted(glob(os.path.join(self.data_path, '*', '*NOISY*')))
        newlwn = len(self.noisy_files)*9//10
        # quick_fix for train val spilt: todo fix
        if not self.deterministic: # train
            self.noisy_files = self.noisy_files[:newlwn]
        else:
            self.noisy_files = self.noisy_files[newlwn:]
        print(f'In total: {len(self.noisy_files)} images in dataset.')
        print(self.noisy_files[0])
        #gt_files = natsorted(glob(os.path.join(self.data_path, '*', '*GT*')))
        #meta_files = natsorted(glob(os.path.join(self.data_path, '*', '*META*')))


    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, index):
        gt_image, noisy_image, py_meta = self.read_rand_crop(index, crop=True) # todo cahnge crop to input value for test
        if not self.deterministic and self.random_flip and random.random() < 0.5:
            noisy_image = torch.flip(noisy_image, (-1,))
            gt_image = torch.flip(gt_image, (-1,))#gt_image[:, ::-1]

        noisy_image = noisy_image*2 -1
        gt_image = gt_image*2 -1
        out_dict = {}
        out_dict['x_T_end'] = noisy_image
        out_dict['noise_Ls'] = py_meta['noise'][0]
        out_dict['noise_Lr'] = py_meta['noise'][1]
        #print(py_meta['noise'])
        return gt_image, out_dict

    def __getitem_old(self, idx):
        img, out_dict = self.get_sample(idx)
        # for example:
        x_T_end = np.random.normal(loc=img, scale=(img+1)*0.1, size=img.shape) + np.random.randn(*img.shape)*0.1
        #out_dict['x_T_end'] = x_T_end.astype(np.float)
        return img, out_dict

    def get_sample(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict

    def read_rand_crop(self, index, crop=True):
        noisy_path = self.noisy_files[index]
        noisy_image = h5py.File(noisy_path, 'r')['x']
        gt_path = noisy_path.replace('NOISY', 'GT')
        gt_image = h5py.File(gt_path, 'r')['x']

        # cropping:
        h0, h1, w0, w1 = self._get_crop_values(noisy_image.shape, crop)
        noisy_image = noisy_image[h0:h1, w0:w1]
        gt_image = gt_image[h0:h1, w0:w1]

        #meta = sio.loadmat(noisy_path.replace('NOISY', 'METADATA'))
        #meta = meta['metadata'][0][0]
        py_meta = extract_metainfo(noisy_path.replace('NOISY', 'METADATA'))


        pattern = py_meta['pattern']
        noisy_image = transform_to_rggb(noisy_image, pattern)
        gt_image = transform_to_rggb(gt_image, pattern)

        noisy_image = torch.from_numpy(noisy_image)
        gt_image = torch.from_numpy(gt_image)

        noisy_image = raw2stack(noisy_image)
        gt_image = raw2stack(gt_image)

        return gt_image, noisy_image, py_meta

    def _get_crop_values(self, shape, crop):
        if not crop:
            return 0, shape[0], 0 , shape[1]
        s = self.resolution
        H, W = shape
        if self.deterministic:
            h0, w0 = 100, 100
        else:
            h0 = np.random.randint(0, H - s)
            w0 = np.random.randint(0, W - s)
        return h0, h0+s, w0, w0+s




''' Sources:
https://github.com/arcchang1236/CA-NoiseGAN/blob/master/test_denoisers.py
https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark
https://github.com/AbdoKamel/sidd-ground-truth-image-estimation
https://github.com/MegEngine/NBNet/blob/main/prepare_data.py
https://github.com/megvii-model/HINet/blob/main/scripts/data_preparation/sidd.py
'''

def source_test_sidd():
    # https://github.com/zhangyi-3/noise-synthesis/blob/main/test.py
    test_data_list = [item for item in os.listdir(root) if
                      int(item.split('_')[1]) in [2, 3, 5] and camera in item.lower()]
    for idx, item in enumerate(test_data_list):
        head = item[:4]
        for tail in ['GT_RAW_010', 'GT_RAW_011']:
            print('processing', idx, item, tail, end=' ')
            mat = utils.open_hdf5(os.path.join(root, item, '%s_%s.MAT' % (head, tail)))
            mat = hdf5_file = h5py.File(filename, 'r')

            gt = np.array(mat['x'], dtype=np.float32)
            mat = utils.open_hdf5(os.path.join(root, item, '%s_%s.MAT' % (head, tail.replace('GT', 'NOISY'))))
            noisy = np.array(mat['x'], dtype=np.float32)

            meta = sio.loadmat(os.path.join(root, item, '%s_%s.MAT' % (head, tail.replace('GT', 'METADATA'))))
            meta = meta['metadata'][0][0]

            # transform to rggb pattern
            py_meta = extract_metainfo(
                os.path.join(root, item, '%s_%s.MAT' % (head, tail.replace('GT', 'METADATA'))))
            pattern = py_meta['pattern']
            noisy = utils.transform_to_rggb(noisy, pattern)
            gt = utils.transform_to_rggb(gt, pattern)

            denoised = forward_patches(model, noisy)

            psnr = peak_signal_noise_ratio(gt, denoised, data_range=1)
            psnr_list.append(psnr)
            print('psnr %.2f' % psnr)

    print('Camera %s, average PSNR %.2f' % (camera, np.mean(psnr_list)))

def extract_metainfo(path='0151_METADATA_RAW_010.MAT'):
    meta = sio.loadmat(path)['metadata']
    mat_vals = meta[0][0]
    mat_keys = mat_vals.dtype.descr

    keys = []
    for item in mat_keys:
        keys.append(item[0])

    py_dict = {}
    for key in keys:
        py_dict[key] = mat_vals[key]

    device = py_dict['Model'][0].lower()
    bitDepth = py_dict['BitDepth'][0][0]
    if 'iphone' in device or bitDepth != 16:
        noise = py_dict['UnknownTags'][-2][0][-1][0][:2]
        iso = py_dict['DigitalCamera'][0, 0]['ISOSpeedRatings'][0][0]
        pattern = py_dict['SubIFDs'][0][0]['UnknownTags'][0][0][1][0][-1][0]
        time = py_dict['DigitalCamera'][0, 0]['ExposureTime'][0][0]

    else:
        noise = py_dict['UnknownTags'][-1][0][-1][0][:2]
        iso = py_dict['ISOSpeedRatings'][0][0]
        pattern = py_dict['UnknownTags'][1][0][-1][0]
        time = py_dict['ExposureTime'][0][0]  # the 0th row and 0th line item

    rgb = ['R', 'G', 'B']
    pattern = ''.join([rgb[i] for i in pattern])

    asShotNeutral = py_dict['AsShotNeutral'][0]
    b_gain, _, r_gain = asShotNeutral

    # only load ccm1
    ccm = py_dict['ColorMatrix1'][0].astype(float).reshape((3, 3))

    return {'device': device,
            'pattern': pattern,
            'iso': iso,
            'noise': noise,
            'time': time,
            'wb': np.array([r_gain, 1, b_gain]),
            'ccm': ccm, }


def transform_to_rggb(img, pattern):
    assert len(img.shape) == 2 and type(img) == np.ndarray

    if pattern.lower() == 'bggr':  # same pattern
        img = np.roll(np.roll(img, 1, axis=1), 1, axis=0)
    elif pattern.lower() == 'rggb':
        pass
    elif pattern.lower() == 'grbg':
        img = np.roll(img, 1, axis=1)
    elif pattern.lower() == 'gbrg':
        img = np.roll(img, 1, axis=0)
    else:
        assert 'no support'

    return img

def raw2stack(image):
    h, w = image.shape
    # if image.is_cuda:
    #     res = torch.cuda.FloatTensor(4, h // 2, w // 2).fill_(0)
    # else:
    #     res = torch.FloatTensor(4, h // 2, w // 2).fill_(0)
    res = torch.zeros(4, h // 2, w // 2, device=image.device, dtype=image.dtype)
    res[0] = image[0::2, 0::2]
    res[1] = image[0::2, 1::2]
    res[2] = image[1::2, 0::2]
    res[3] = image[1::2, 1::2]
    return res

def stack2raw(var):
    _, h, w = var.shape
    if var.is_cuda:
        res = torch.cuda.FloatTensor(h * 2, w * 2)
    else:
        res = torch.FloatTensor(h * 2, w * 2)
    res[0::2, 0::2] = var[0]
    res[0::2, 1::2] = var[1]
    res[1::2, 0::2] = var[2]
    res[1::2, 1::2] = var[3]
    return

################## break

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
