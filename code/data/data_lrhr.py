import os

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random

import cv2
import random
import glob
import random


class DS(Dataset):
    def __init__(self, lr_path, hr_path, hr_size, scale):
        self.samples = []
        for hr_path, _, fnames in sorted(os.walk(hr_path)):
            for fname in sorted(fnames):
                path = os.path.join(hr_path, fname)
                self.samples.append(path)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + hr_path)
        self.hr_size = hr_size
        self.scale = scale
        self.lr_path = lr_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # getting hr image
        hr_path = self.samples[index]
        hr_image = cv2.imread(hr_path)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # getting lr image
        lr_path = os.path.join(self.lr_path, os.path.basename(hr_path))
        lr_image = cv2.imread(lr_path)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # checking for hr_size limitation
        if hr_image.shape[0] > self.hr_size or hr_image.shape[1] > self.hr_size:
          # image too big, random crop
          random_pos1 = random.randint(0,hr_image.shape[0]-self.hr_size)
          random_pos2 = random.randint(0,hr_image.shape[0]-self.hr_size)

          image_hr = hr_image[random_pos1:random_pos1+self.hr_size, random_pos2:random_pos2+self.hr_size]
          image_lr = lr_image[int(random_pos1/self.scale):int((random_pos2+self.hr_size)/self.scale), int(random_pos2/self.scale):int((random_pos2+self.hr_size)/self.scale)]

        # to tensor
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1)/255
        lr_image = torch.from_numpy(lr_image).permute(2, 0, 1)/255

        return lr_image, hr_image


class DS_val(Dataset):
    def __init__(self, lr_path, hr_path):
        self.samples = []
        for hr_path, _, fnames in sorted(os.walk(hr_path)):
            for fname in sorted(fnames):
                path = os.path.join(hr_path, fname)
                self.samples.append(path)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + hr_path)

        self.lr_path = lr_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # getting hr image
        hr_path = self.samples[index]
        hr_image = cv2.imread(hr_path)

        # getting lr image
        lr_path = os.path.join(self.lr_path, os.path.basename(hr_path))
        lr_image = cv2.imread(lr_path)

        # to tensor
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1)/255
        lr_image = torch.from_numpy(lr_image).permute(2, 0, 1)/255

        return lr_image, hr_image, lr_path
