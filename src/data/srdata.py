import os
from src.data import common

import numpy as np
import cv2

import torch
import torch.utils.data as data


class SRData(data.Dataset):

    def __init__(self, args, train=True):
        self.args = args
        self.train = train

        self.scale = args.scale
        self.idx_scale = 0
        self.repeat = args.repeat
        self._set_filesystem(args.dir_data)

        self.images_hr, self.images_lr = self._scan()

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + "/dataset"
        if self.train:
            self.dir_hr = os.path.join(self.apath, 'HR_train')
            self.dir_lr = os.path.join(self.apath, f'LR_train/x{self.scale}')
        else:
            self.dir_hr = os.path.join(self.apath, 'HR_test')
            self.dir_lr = os.path.join(self.apath, f'LR_test/x{self.scale}')
        self.ext = '.png'

    def _scan(self):
        list_hr = sorted(os.listdir(self.dir_hr))
        list_lr = sorted(os.listdir(self.dir_lr))

        for i in range(len(list_hr)):
            list_hr[i] = os.path.join(self.dir_hr, list_hr[i])
            list_lr[i] = os.path.join(self.dir_lr, list_lr[i])

        return list_hr, list_lr

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[idx]
        hr = self.images_hr[idx]
        filename = os.path.basename(hr)
        lr = cv2.imread(lr)
        hr = cv2.imread(hr)

        return lr, hr, filename

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _get_patch(self, lr, hr):
        patch_size = self.args.patch_size

        if self.train:
            lr, hr = common.get_patch(lr, hr, patch_size, self.scale)
            lr, hr = common.augment([lr, hr])

        else:
            ih, iw = lr.shape[0:2]
            ih = min(ih, 1000)
            iw = min(iw, 800)
            lr = lr[:ih, :iw]
            hr = hr[0:ih * self.scale, 0:iw * self.scale]

        return lr, hr

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr])
        return lr_tensor, hr_tensor, filename
