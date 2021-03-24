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
    def __init__(self, root, transform=None, size=256, batch_size_DL = 3, scale=4, image_size=400, amount_tiles=3):
        self.samples = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                self.samples.append(path)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root)

        self.transform = transform

        #self.size = size
        self.image_size = image_size # how big one tile is
        self.scale = scale
        self.batch_size = batch_size_DL
        self.interpolation_method = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        self.amount_tiles = amount_tiles

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        sample = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)


        pos_total = []

        self.total_size = 0 # the current amount of images that got a random position

        while True:
          # determine random position
          x_rand = random.randint(0,self.amount_tiles-1)
          y_rand = random.randint(0,self.amount_tiles-1)

          pos_rand = [x_rand, y_rand]

          if (pos_rand in pos_total) != True:
            pos_total.append(pos_rand)
            self.total_size += 1

          # return batchsize
          if self.total_size == self.batch_size:
            break

        self.total_size = 0 # counter for making sure array gets appended if processed images > 1

        for i in pos_total:
          # creating sample if for start
          if self.total_size == 0:
            # cropping from hr image
            image_hr = sample[i[0]*self.image_size:(i[0]+1)*self.image_size, i[1]*self.image_size:(i[1]+1)*self.image_size]
            # creating lr on the fly
            #image_lr = cv2.resize(image_hr, (int(self.image_size/self.scale), int(self.image_size/self.scale)), interpolation=random.choice(self.interpolation_method))
            image_lr = cv2.resize(image_hr, (int(self.image_size/self.scale), int(self.image_size/self.scale)), interpolation=random.choice(self.interpolation_method))

            """
            print("-----------------------")
            print(i[0]*(self.image_size/self.scale))
            print((i[0]+1)*(self.image_size/self.scale))

            print(i[1]*(self.image_size/self.scale))
            print((i[1]+1)*(self.image_size/self.scale))
            """
            #image_lr = image_lr[i[0]*(self.image_size/self.scale):(i[0]+1)*(self.image_size/self.scale), i[1]*(self.image_size/self.scale):(i[1]+1)*(self.image_size/self.scale)]


            # creating torch tensor
            image_hr = torch.from_numpy(image_hr).unsqueeze(2).permute(2, 0, 1).unsqueeze(0)/255
            image_lr = torch.from_numpy(image_lr).unsqueeze(2).permute(2, 0, 1).unsqueeze(0)/255


            # if edges are required
            """
            grayscale = cv2.cvtColor(np.array(sample_add), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(grayscale,100,150)
            grayscale = torch.from_numpy(grayscale).unsqueeze(0)/255
            edges = torch.from_numpy(edges).unsqueeze(0)
            """

            self.total_size += 1
          else:
            # cropping from hr image
            image_hr2 = sample[i[0]*self.image_size:(i[0]+1)*self.image_size, i[1]*self.image_size:(i[1]+1)*self.image_size]
            # creating lr on the fly
            #image_lr2 = cv2.resize(image_hr2, (int(self.image_size/self.scale), int(self.image_size/self.scale)), interpolation=random.choice(self.interpolation_method))
            #image_lr2 = image_lr2[i[0]*(self.image_size/self.scale):(i[0]+1)*(self.image_size/self.scale), i[1]*(self.image_size/self.scale):(i[1]+1)*(self.image_size/self.scale)]
            image_lr2 = cv2.resize(image_hr2, (int(self.image_size/self.scale), int(self.image_size/self.scale)), interpolation=random.choice(self.interpolation_method))



            # if edges are required
            """
            grayscale = cv2.cvtColor(np.array(sample_add2), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(grayscale,100,150)
            grayscale = torch.from_numpy(grayscale).unsqueeze(0)/255
            edges = torch.from_numpy(edges).unsqueeze(0)
            """
            # creating torch tensor
            image_hr2 = torch.from_numpy(image_hr2).unsqueeze(2).permute(2, 0, 1).unsqueeze(0)/255
            image_hr = torch.cat((image_hr, image_hr2), dim=0)

            image_lr2 = torch.from_numpy(image_lr2).unsqueeze(2).permute(2, 0, 1).unsqueeze(0)/255
            image_lr = torch.cat((image_lr, image_lr2), dim=0)

        return image_lr, image_hr


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
        hr_image = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)

        # getting lr image
        lr_path = os.path.join(self.lr_path, os.path.basename(hr_path))
        lr_image = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)



        hr_image = torch.from_numpy(hr_image).unsqueeze(2).permute(2, 0, 1).unsqueeze(0)/255
        lr_image = torch.from_numpy(lr_image).unsqueeze(2).permute(2, 0, 1).unsqueeze(0)/255


        return lr_image, hr_image, lr_path
