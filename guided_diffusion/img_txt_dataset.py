import math
import os.path

import random
import torch

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import load
from os.path import dirname, join
import clip

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    clip_file_path=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond and False:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        clip_file_path=clip_file_path,
        deterministic=deterministic
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


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


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        clip_file_path=None,
        deterministic=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        #clip_file_path = join(dirname(dirname(self.local_images[0])), 'thumbnails128x128_ViT-B32_dict.pt')
        #print(clip_file_path, os.path.isfile(clip_file_path))
        assert clip_file_path is not None, f'path: {clip_file_path}'
        self.clip_file_path = clip_file_path
        self.clip_data = load(clip_file_path, map_location='cpu')
        self.deterministic = deterministic

        model, preprocess = clip.load("ViT-B/32", device='cpu')
        model.eval()
        with torch.no_grad():
            text = clip.tokenize(["a face with long hair"]).to('cpu') # a face with long hair
            self.y0 = model.encode_text(text)[0]
            text2 = clip.tokenize(["face"]).to('cpu')
            self.x0 = model.encode_text(text2)[0]
            self.delta0 = self.y0 - self.x0

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        idx=2 # 21=Mying
        img, out_dict = self.get_sample(idx)
        out_dict['clip_feat_img_save'] = out_dict['clip_feat'][0].clone()
        #print(out_dict['spat_feat'])
        # add img2:
        '''
        if not self.deterministic:
            if random.random() < 0.15:
                idx2 = idx
                idx2_data = img, out_dict
            else:
                idx2 = random.randint(0, len(self)-1)#
                idx2_data = self.get_sample(idx2)
        else:
            idx2 = idx if idx<4 else idx-1
            idx2_data = (img, out_dict) if idx<4 else self.get_sample(idx-1)
        '''

        #img2, out_dict2 = idx2_data
        out_dict['img2'] = img # SOURCE IMAGE To CONCAT with the noise

        #out_dict['clip_feat2'] = out_dict2['clip_feat']
        out_dict['clip_feat2'] = self.x0 # source point x0 want to change to y0
        out_dict['clip_feat'] = self.y0 # target point y0. the direction is Delta0=target-source
        return_diff = True
        if return_diff:
            out_dict['img'] = img # target is missing in unsup
            img = np.zeros_like(img) # img - img2 starting at zero since target is missing in unsup

        return img, out_dict

    def get_sample(self, idx):
        path = self.local_images[idx]
        # print(self.local_images[idx], idx)
        # print(self.local_images[idx-1], idx-1)
        # print(self.local_images[idx+1], idx+1)
        # #s = [x for x in self.local_images if 'm.png' in x][0]
        # print(len(self.local_images))
        #print(self.local_images.index(s))
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)
        img_flipped = self.random_flip and random.random() < 0.5
        if img_flipped:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if 'caleba' in self.clip_file_path:
            out_dict['clip_feat'] = self.clip_data[bf.basename(path)]
        else:
            out_dict['clip_feat'] = self.clip_data[bf.basename(path)][int(img_flipped)]
        return np.transpose(arr, [2, 0, 1]), out_dict



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
