import torch as th
import numpy as np
import os

from glob import glob
from PIL import Image
from random import random

import torchvision.transforms as tvf


class ImageDataset(th.utils.data.Dataset):
    def __init__(self, path, ext, img_params, tensor_params):
        self.list = np.sort(glob(os.path.join(path, f"*.{ext}")))
        self.img_transform = ImageTransform(**img_params)
        self.tensor_transform = TensorTransform(**tensor_params)

    def __getitem__(self, idx):
        image = Image.open(self.list[idx]).convert("RGB")
        image = self.img_transform(image)
        return self.tensor_transform(image)

    def __len__(self):
        return len(self.list)


class ImageTransform:
    def __init__(self, size=256, rc2cc=None,
                 hflip=None, vflip=None, jitter=None):
        t_s = []
        if size is not None:
            t_s.append(MinResize(size))
            t_s.append(tvf.Resize(size))
        if rc2cc is not None:
            t_s.append(RandomCrop(size, rc2cc))
        if hflip is not None:
            t_s.append(tvf.RandomHorizontalFlip())
        if vflip is not None:
            t_s.append(tvf.RandomVerticalFlip())
        if jitter is not None:
            t_s.append(tvf.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.5, hue=(-0.3, 0.3)))
        self.t = tvf.Compose(t_s)

    def __call__(self, image):
        return self.t(image)


class TensorTransform:
    def __init__(self, norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), erase=None):
        t_s = [tvf.ToTensor()]
        if norm is not None:
            t_s.append(tvf.Normalize(*norm))
        if erase is not None:
            t_s.append(tvf.RandomErasing(erase))
        self.t = tvf.Compose(t_s)

    def __call__(self, image):
        return self.t(image)


class RandomCrop:
    def __init__(self, size, rc2cc=0.):
        self.rc2cc = rc2cc
        self.size = size
        self.rrc = tvf.RandomResizedCrop(size,
                                         scale=(0.5, 1.0),
                                         ratio=(0.98, 1.02))
        self.cc = tvf.CenterCrop(size)

    def __call__(self, x):
        if random() < self.rc2cc:
            return self.rrc(x)
        else:
            return self.cc(x)


class MinResize:
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, image: Image):
        if max(*image.size) < self.min_size:
            image = tvf.functional.resize(image, self.min_size)
        return image
