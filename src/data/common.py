import random
import numpy as np
import cv2

import torch
from torchvision import transforms


def get_patch(img_in, img_tar, patch_size, scale):
    ih, iw = img_in.shape[:2]

    tp = patch_size
    ip = patch_size // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar


def set_channel(l, n_channel):

    def _set_channel(img):
        if img.ndim == 2:  # 如果图像是单通道的
            img = np.expand_dims(img, axis=2)  # 扩展为 (H, W, 1)

        c = img.shape[2]  # 获取当前图像的通道数
        if n_channel == 1 and c == 3:
            # 将图像从 BGR 转换为 YCrCb 并提取 Y 通道
            img = np.expand_dims(
                cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            # 将单通道图像复制扩展为三通道
            img = np.concatenate([img] * n_channel, 2)

        return img

    # 对列表中的每个图像应用 _set_channel 函数
    return [_set_channel(_l) for _l in l]


def np2Tensor(l):

    def _np2Tensor(img: np.ndarray):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        return tensor

    return [_np2Tensor(_l) for _l in l]

def add_gaussian_noise(image, mean=0, sigma=3):
    noisy_image = image.astype(np.float32)

    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)

    noisy_image = noisy_image + gauss

    noisy_image = np.clip(noisy_image, 0, 255)

    noisy_image = noisy_image.astype(np.uint8)
    
    return noisy_image

def add_gaussian_blue(image):
    return cv2.GaussianBlur(image, (3,3), 1)

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    blur = rot and random.random() < 0.5
    noise = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        if blur: img = add_gaussian_blue(img)
        if noise: img = add_gaussian_noise(img)

        return img

    return [_augment(_l) for _l in l]
