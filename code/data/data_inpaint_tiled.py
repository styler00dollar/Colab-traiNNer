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
    def __init__(self, root, transform=None, size=256):
        self.samples = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                self.samples.append(path)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root)

        self.transform = transform
        self.mask_dir = '/content/masks'
        self.files = glob.glob(self.mask_dir + '/**/*.png', recursive=True)
        files_jpg = glob.glob(self.mask_dir + '/**/*.jpg', recursive=True)
        self.files.extend(files_jpg)

        self.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        #sample = Image.open(sample_path).convert('RGB')
        sample = cv2.imread(sample_path)


        x_rand = random.randint(0,15)
        y_rand = random.randint(0,15)

        sample = sample[x_rand*256:(x_rand+1)*256, y_rand*256:(y_rand+1)*256]
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        #sample = torch.from_numpy(sample)

        #if self.transform:
        #    sample = self.transform(sample)

        # if edges are required
        grayscale = cv2.cvtColor(np.array(sample), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(grayscale,100,150)
        grayscale = torch.from_numpy(grayscale).unsqueeze(0)/255
        edges = torch.from_numpy(edges).unsqueeze(0)

        if random.uniform(0, 1) < 0.5:
          # generating mask automatically with 50% chance
          mask = DS.random_mask(height=self.size, width=self.size)
          mask = torch.from_numpy(mask)

        else:
          # load random mask from folder
          mask = cv2.imread(random.choice([x for x in self.files]), cv2.IMREAD_UNCHANGED)
          mask = cv2.resize(mask, (self.size,self.size), interpolation=cv2.INTER_NEAREST)

          # flip mask randomly
          if 0.3 < random.uniform(0, 1) <= 0.66:
            mask = np.flip(mask, axis=0)
          elif 0.66 < random.uniform(0, 1) <= 1:
            mask = np.flip(mask, axis=1)

          mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)/255

        sample = torch.from_numpy(sample).permute(2, 0, 1)/255
        #sample = transforms.ToTensor()(sample)

        # apply mask
        #print(sample.shape)
        #print(mask.shape)
        masked = sample * mask
        return masked, mask, sample

        # EdgeConnect
        #return masked, mask, sample, edges, grayscale

        # PRVS
        #return masked, mask, sample, edges


    @staticmethod
    def random_mask(height=256, width=256,
                    min_stroke=1, max_stroke=4,
                    min_vertex=1, max_vertex=12,
                    min_brush_width_divisor=16, max_brush_width_divisor=10):
        mask = np.ones((height, width))

        min_brush_width = height // min_brush_width_divisor
        max_brush_width = height // max_brush_width_divisor
        max_angle = 2*np.pi
        num_stroke = np.random.randint(min_stroke, max_stroke+1)
        average_length = np.sqrt(height*height + width*width) / 8

        for _ in range(num_stroke):
            num_vertex = np.random.randint(min_vertex, max_vertex+1)
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)

            for _ in range(num_vertex):
                angle = np.random.uniform(max_angle)
                length = np.clip(np.random.normal(average_length, average_length//2), 0, 2*average_length)
                brush_width = np.random.randint(min_brush_width, max_brush_width+1)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 0., brush_width)

                start_x, start_y = end_x, end_y
        if np.random.random() < 0.5:
            mask = np.fliplr(mask)
        if np.random.random() < 0.5:
            mask = np.flipud(mask)
        return mask.reshape((1,)+mask.shape).astype(np.float32)



class DS_green_from_mask(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                self.samples.append(path)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root)

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        #sample = Image.open(sample_path).convert('RGB')
        sample = cv2.imread(sample_path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        # if edges are required
        grayscale = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(grayscale,100,150)
        grayscale = torch.from_numpy(grayscale).unsqueeze(0)
        edges = torch.from_numpy(edges).unsqueeze(0)

        green_mask = 1-np.all(sample == [0,255,0], axis=-1).astype(int)
        green_mask = torch.from_numpy(green_mask).unsqueeze(0)
        sample = torch.from_numpy(sample.astype(np.float32)).permute(2, 0, 1)/255
        sample = sample * green_mask

        # train_batch[0] = masked
        # train_batch[1] = mask
        # train_batch[2] = path
        return sample, green_mask, sample_path

        # EdgeConnect
        #return sample, green_mask, sample_path, edges, grayscale

        # PRVS
        #return sample, green_mask, sample_path, edges
