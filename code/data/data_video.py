import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import *
import torchvision
from PIL import Image
import random
import cv2
import torchvision.transforms.functional as TF
import glob

class VimeoTriplet(Dataset):
    def __init__(self, data_root):
        upper_folders = glob.glob(data_root + "/*/")

        self.samples = []

        # getting subfolders of folders
        for f in upper_folders:
          self.samples.append(glob.glob(f + "/*/"))

        # for subfolders
        self.samples = [item for sublist in self.samples for item in sublist]
        #self.samples = upper_folders

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        imgpaths = [self.samples[index] + '/im1.png', self.samples[index] + '/im2.png', self.samples[index] + '/im3.png']
        # Load images
        img1 = cv2.imread(imgpaths[0])
        img2 = cv2.imread(imgpaths[1])
        img3 = cv2.imread(imgpaths[2])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        """
        if random.random() >= 0.5:
            img1 = Image.fromarray(img1[0:0+256, 0:0+256])
            img2 = Image.fromarray(img2[0:0+256, 0:0+256])
            img3 = Image.fromarray(img3[0:0+256, 0:0+256])
        else:
            img1 = Image.fromarray(img1[0:0+256, 192:256+256])
            img2 = Image.fromarray(img2[0:0+256, 192:256+256])
            img3 = Image.fromarray(img3[0:0+256, 192:256+256])
        """
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        img3 = cv2.resize(img3, (256, 256))

        # Data augmentation COLOR_YUV_I4202RGB
        img1 = self.transforms(img1)
        img2 = self.transforms(img2)
        img3 = self.transforms(img3)
        if random.random() >= 0.5:
            img1, img3 = img3, img1

        #imgs = [img1, img2, img3]

        return img1, img3, img2


class VimeoTriplet_val(Dataset):
    def __init__(self, data_root):
        upper_folders = glob.glob(data_root + "/*/")

        self.samples = []

        # getting subfolders of folders
        for f in upper_folders:
          self.samples.append(glob.glob(f + "/*/"))


        # this is for subfolders
        #self.samples = [item for sublist in self.samples for item in sublist]
        self.samples = upper_folders

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        imgpaths = [self.samples[index] + '/im1.jpg', self.samples[index] + '/im2.jpg', self.samples[index] + '/im3.jpg']
        # Load images
        img1 = cv2.imread(imgpaths[0])
        img2 = cv2.imread(imgpaths[1])
        img3 = cv2.imread(imgpaths[2])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        """
        if random.random() >= 0.5:
            img1 = Image.fromarray(img1[0:0+256, 0:0+256])
            img2 = Image.fromarray(img2[0:0+256, 0:0+256])
            img3 = Image.fromarray(img3[0:0+256, 0:0+256])
        else:
            img1 = Image.fromarray(img1[0:0+256, 192:256+256])
            img2 = Image.fromarray(img2[0:0+256, 192:256+256])
            img3 = Image.fromarray(img3[0:0+256, 192:256+256])
        """
        img1 = cv2.resize(img1, (1280, 720))
        img2 = cv2.resize(img2, (1280, 720))
        img3 = cv2.resize(img3, (1280, 720))

        # Data augmentation COLOR_YUV_I4202RGB
        img1 = self.transforms(img1)
        img2 = self.transforms(img2)
        img3 = self.transforms(img3)
        if random.random() >= 0.5:
            img1, img3 = img3, img1

        #imgs = [img1, img2, img3]
        imgs = [img1, img3]

        path = '_'.join(os.path.normpath(imgpaths[1]).split(os.sep)[-3:-1]) + ".png"
        return imgs, img2, path
