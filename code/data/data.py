import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

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
import torchvision.transforms as transforms

if cfg['datasets']['train']['mode'] == "DS_inpaint_TF" or cfg['datasets']['train']['mode'] == "DS_svg_TF":
  from tfrecord.torch.dataset import TFRecordDataset
  import io

if cfg['datasets']['train']['mode'] == "DS_svg_TF":
  from cairosvg import svg2png
  from io import BytesIO

if cfg['datasets']['train']['loading_backend'] == "PIL":
  import pillow_avif

def random_mask(height=256, width=256,
                    min_stroke=1, max_stroke=10,
                    min_vertex=1, max_vertex=15,
                    min_brush_width_divisor=12, max_brush_width_divisor=5):
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



class DS_inpaint(Dataset):
    def __init__(self, root, mask_dir, HR_size=256):
        self.samples = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if ".png" in path or ".jpg" in path or ".webp" in path:
                  self.samples.append(path)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root)

        self.mask_dir = mask_dir
        self.files = glob.glob(self.mask_dir + '/**/*.png', recursive=True)
        files_jpg = glob.glob(self.mask_dir + '/**/*.jpg', recursive=True)
        self.files.extend(files_jpg)

        self.HR_size = HR_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        sample = cv2.imread(sample_path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        # if edges are required
        if cfg['network_G']['netG'] == 'EdgeConnect' or cfg['network_G']['netG'] == 'PRVS' or cfg['network_G']['netG'] == 'CTSDG':
          grayscale = cv2.cvtColor(np.array(sample), cv2.COLOR_RGB2GRAY)
          edges = cv2.Canny(grayscale,100,150)
          grayscale = torch.from_numpy(grayscale).unsqueeze(0)/255
          edges = torch.from_numpy(edges).unsqueeze(0).type(torch.float)

        if random.uniform(0, 1) < 0.5:
          # generating mask automatically with 50% chance
          mask = random_mask(height=self.HR_size, width=self.HR_size)
          mask = torch.from_numpy(mask)

        else:
          # load random mask from folder
          mask = cv2.imread(random.choice([x for x in self.files]), cv2.IMREAD_UNCHANGED)
          mask = cv2.resize(mask, (self.HR_size,self.HR_size), interpolation=cv2.INTER_NEAREST)

          # since white = masked area, invert mask
          mask = 255 - mask

          # flip mask randomly
          if 0.3 < random.uniform(0, 1) <= 0.66:
            mask = np.flip(mask, axis=0)
          elif 0.66 < random.uniform(0, 1) <= 1:
            mask = np.flip(mask, axis=1)

          mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)/255

        sample = torch.from_numpy(sample).permute(2, 0, 1)/255

        # chance of the mask being inverted
        if random.uniform(0, 1) < cfg['datasets']['train']['mask_invert_ratio']:
          mask = 1 - mask

        # apply mask
        masked = sample * mask

        # EdgeConnect
        if cfg['network_G']['netG'] == 'EdgeConnect':
          return masked, mask, sample, edges, grayscale

        # PRVS
        elif cfg['network_G']['netG'] == 'PRVS' or cfg['network_G']['netG'] == 'CTSDG':
          return masked, mask, sample, edges

        else:
          return masked, mask, sample

class DS_inpaint_val(Dataset):
    def __init__(self, root):
        self.samples = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if ".png" in path or ".jpg" in path or ".webp" in path:
                  self.samples.append(path)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        #sample = Image.open(sample_path).convert('RGB')
        sample = cv2.imread(sample_path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        # if edges are required
        if cfg['network_G']['netG'] == 'EdgeConnect' or cfg['network_G']['netG'] == 'PRVS' or cfg['network_G']['netG'] == 'CTSDG':
          grayscale = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
          edges = cv2.Canny(grayscale,100,150)
          grayscale = torch.from_numpy(grayscale).unsqueeze(0)
          edges = torch.from_numpy(edges).unsqueeze(0).type(torch.float)

        green_mask = 1-np.all(sample == [0,255,0], axis=-1).astype(int)
        green_mask = torch.from_numpy(green_mask.astype(np.float32)).unsqueeze(0)
        sample = torch.from_numpy(sample.astype(np.float32)).permute(2, 0, 1)/255
        sample = sample * green_mask

        # train_batch[0] = masked
        # train_batch[1] = mask
        # train_batch[2] = path

        # EdgeConnect
        if cfg['network_G']['netG'] == 'EdgeConnect':
          return sample, green_mask, sample_path, edges, grayscale

        # PRVS
        elif cfg['network_G']['netG'] == 'PRVS' or cfg['network_G']['netG'] == 'CTSDG':
          return sample, green_mask, sample_path, edges

        else:
          return sample, green_mask, sample_path

# DFDNet
def get_part_location(landmark_path, imgname, downscale=1):
    Landmarks = []
    path = os.path.join(landmark_path, str(imgname) + '.txt')
    with open(path,'r') as f:
        for line in f:
            tmp = [np.float(i) for i in line.split(' ') if i != '\n']
            Landmarks.append(tmp)
    Landmarks = np.array(Landmarks)/downscale # 512 * 512

    Map_LE = list(np.hstack((range(17,22), range(36,42))))
    Map_RE = list(np.hstack((range(22,27), range(42,48))))
    Map_NO = list(range(29,36))
    Map_MO = list(range(48,68))
    #left eye
    Mean_LE = np.mean(Landmarks[Map_LE],0)
    L_LE = np.max((np.max(np.max(Landmarks[Map_LE],0) - np.min(Landmarks[Map_LE],0))/2,16))
    Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
    #right eye
    Mean_RE = np.mean(Landmarks[Map_RE],0)
    L_RE = np.max((np.max(np.max(Landmarks[Map_RE],0) - np.min(Landmarks[Map_RE],0))/2,16))
    Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
    #nose
    Mean_NO = np.mean(Landmarks[Map_NO],0)
    L_NO = np.max((np.max(np.max(Landmarks[Map_NO],0) - np.min(Landmarks[Map_NO],0))/2,16))
    Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
    #mouth
    Mean_MO = np.mean(Landmarks[Map_MO],0)
    L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16))

    Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
    return Location_LE, Location_RE, Location_NO, Location_MO



class DS_lrhr(Dataset):
    def __init__(self, lr_path, hr_path, hr_size = 256, scale = 4, transform = None):
        self.samples = []
        for hr_path, _, fnames in sorted(os.walk(hr_path)):
            for fname in sorted(fnames):
                path = os.path.join(hr_path, fname)
                if ".png" in path or ".jpg" in path or ".webp" in path:
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
          random_pos2 = random.randint(0,hr_image.shape[1]-self.hr_size)


          hr_image = hr_image[random_pos1:random_pos1+self.hr_size, random_pos2:random_pos2+self.hr_size]
          lr_image = lr_image[int(random_pos1/self.scale):int((random_pos1/self.scale)+self.hr_size/self.scale),
                              int(random_pos2/self.scale):int((random_pos2/self.scale)+self.hr_size/self.scale)]

        # to tensor
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1)/255
        lr_image = torch.from_numpy(lr_image).permute(2, 0, 1)/255

        # if generator is DFDNet, change image range to [-1,1] and also pass landmarks
        if cfg['network_G']['netG'] == 'DFDNet':
          hr_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(hr_image)
          lr_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(lr_image)

          landmarks = get_part_location(landmark_path=cfg['network_G']['landmarkpath'], imgname=os.path.basename(hr_path), downscale=1)
          return 0, lr_image, hr_image, landmarks
        else:
          return 0, lr_image, hr_image


class DS_lrhr_val(Dataset):
    def __init__(self, lr_path, hr_path):
        self.samples = []
        for hr_path, _, fnames in sorted(os.walk(hr_path)):
            for fname in sorted(fnames):
                path = os.path.join(hr_path, fname)
                if ".png" in path or ".jpg" in path or ".webp" in path:
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
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # getting lr image
        lr_path = os.path.join(self.lr_path, os.path.basename(hr_path))
        lr_image = cv2.imread(lr_path)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # to tensor
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1)/255
        lr_image = torch.from_numpy(lr_image).permute(2, 0, 1)/255

        # if generator is DFDNet, change image range to [-1,1] and also pass landmarks
        if cfg['network_G']['netG'] == 'DFDNet':
          hr_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(hr_image)
          lr_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(lr_image)

          landmarks = get_part_location(landmark_path=cfg['network_G']['val_landmarkpath'], imgname=os.path.basename(hr_path), downscale=1)
          return lr_image, hr_image, lr_path, landmarks
        else:
          return lr_image, hr_image, lr_path



class DS_inpaint_TF(Dataset):
    def __init__(self):
      tfrecord_path = cfg['datasets']['train']['tfrecord_path']
      self.mask_dir = cfg['datasets']['train']['masks']
      self.mask_files = glob.glob(self.mask_dir + '/**/*.png', recursive=True)

      self.HR_size = cfg['datasets']['train']['HR_size']
      #self.batch_size = cfg['datasets']['train']['batch_size']

      self.dataset = TFRecordDataset(tfrecord_path, None)
      self.loader = iter(torch.utils.data.DataLoader(self.dataset, batch_size=1))

    def __len__(self):
        # iterator is infinite and does not have a length, hotfix
        return cfg['datasets']['train']['amount_files']

    def __getitem__(self, index):
        data = next(self.loader)

        if cfg['datasets']['train']['loading_backend'] == "OpenCV":
          nparr = np.fromstring(np.array(data['data']), np.uint8)
          sample = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif cfg['datasets']['train']['loading_backend'] == "PIL":
          sample = Image.open(io.BytesIO(np.array(data['data'])))
          sample = np.array(sample)

        # if edges are required
        if cfg['network_G']['netG'] == 'EdgeConnect' or cfg['network_G']['netG'] == 'PRVS' or cfg['network_G']['netG'] == 'CTSDG':
          grayscale = cv2.cvtColor(np.array(sample), cv2.COLOR_RGB2GRAY)
          edges = cv2.Canny(grayscale,100,150)
          grayscale = torch.from_numpy(grayscale).unsqueeze(0)/255
          edges = torch.from_numpy(edges).unsqueeze(0).type(torch.float)

        if random.uniform(0, 1) < 0.5:
          # generating mask automatically with 50% chance
          mask = random_mask(height=self.HR_size, width=self.HR_size)
          mask = torch.from_numpy(mask)

        else:
          # load random mask from folder
          mask = cv2.imread(random.choice([x for x in self.mask_files]), cv2.IMREAD_UNCHANGED)
          mask = cv2.resize(mask, (self.HR_size,self.HR_size), interpolation=cv2.INTER_NEAREST)
          # since white = masked area, invert mask
          mask = 255 - mask

          # flip mask randomly
          if 0.3 < random.uniform(0, 1) <= 0.66:
            mask = np.flip(mask, axis=0)
          elif 0.66 < random.uniform(0, 1) <= 1:
            mask = np.flip(mask, axis=1)

          mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)/255

        sample = torch.from_numpy(sample).permute(2, 0, 1)/255

        # chance of the mask being inverted
        if random.uniform(0, 1) < cfg['datasets']['train']['mask_invert_ratio']:
          mask = 1 - mask

        # apply mask
        masked = sample * mask

        # EdgeConnect
        if cfg['network_G']['netG'] == 'EdgeConnect':
          return masked, mask, sample, edges, grayscale

        # PRVS
        elif cfg['network_G']['netG'] == 'PRVS' or cfg['network_G']['netG'] == 'CTSDG':
          return masked, mask, sample, edges

        else:
          return masked, mask, sample




class DS_svg_TF(Dataset):
    def __init__(self):
      tfrecord_path = cfg['datasets']['train']['tfrecord_path']

      self.HR_size = cfg['datasets']['train']['HR_size']
      #self.batch_size = cfg['datasets']['train']['batch_size']

      self.dataset = TFRecordDataset(tfrecord_path, None)
      self.loader = iter(torch.utils.data.DataLoader(self.dataset, batch_size=1))

    def __len__(self):
        # iterator is infinite and does not have a length, hotfix
        return cfg['datasets']['train']['amount_files']

    def __getitem__(self, index):
        data = next(self.loader)

        dpi = 300
        output_width = 256
        output_height = 256

        png = svg2png(bytestring=np.fromstring(np.array(data['data']), np.uint8).tobytes(), dpi = dpi, output_width=output_width, output_height=output_height)
        
        # background fix
        hr_image = Image.open(BytesIO(png)).convert('RGBA')
        background = Image.new('RGBA', hr_image.size, (255,255,255))
        hr_image = Image.alpha_composite(background, hr_image)
        hr_image = hr_image.convert('RGB')

        # resize
        lr_image = hr_image.resize((64,64), Image.ANTIALIAS)

        # to tensor
        lr_image = transforms.ToTensor()(lr_image) #.unsqueeze_(0)
        hr_image = transforms.ToTensor()(hr_image) #.unsqueeze_(0)

        #from torchvision.utils import save_image
        #save_image(hr_image*255, 'hr_image.png')

        return 0, lr_image, hr_image