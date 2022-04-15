import yaml
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
from .augmentation import transforms
import random

INTERP_MAP = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'AREA': cv2.INTER_AREA,
              'BICUBIC': cv2.INTER_CUBIC, 'LANCZOS': cv2.INTER_LANCZOS4}

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
with open("aug_config.yaml", "r") as ymlfile:
    augcfg = yaml.safe_load(ymlfile)

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
    def __init__(self, root, mask_dir, hr_size=256):
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

        self.HR_size = hr_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        sample = cv2.imread(sample_path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        # if edges are required
        if (cfg['network_G']['netG'] == 'EdgeConnect'
                or cfg['network_G']['netG'] == 'PRVS'
                or cfg['network_G']['netG'] == 'CTSDG'):
            grayscale = cv2.cvtColor(np.array(sample), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(grayscale, 100, 150)
            grayscale = torch.from_numpy(grayscale).unsqueeze(0)/255
            edges = torch.from_numpy(edges).unsqueeze(0).type(torch.float)

        if random.uniform(0, 1) < 0.5:
            # generating mask automatically with 50% chance
            mask = random_mask(height=self.HR_size, width=self.HR_size)
            mask = torch.from_numpy(mask)

        else:
            # load random mask from folder
            mask = cv2.imread(random.choice([x for x in self.files]), cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask, (self.HR_size, self.HR_size), interpolation=cv2.INTER_NEAREST)

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
        elif (cfg['network_G']['netG'] == 'PRVS'
              or cfg['network_G']['netG'] == 'CTSDG'):
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
        # sample = Image.open(sample_path).convert('RGB')
        sample = cv2.imread(sample_path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        # if edges are required
        if (cfg['network_G']['netG'] == 'EdgeConnect'
                or cfg['network_G']['netG'] == 'PRVS'
                or cfg['network_G']['netG'] == 'CTSDG'):
            grayscale = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(grayscale, 100, 150)
            grayscale = torch.from_numpy(grayscale).unsqueeze(0)
            edges = torch.from_numpy(edges).unsqueeze(0).type(torch.float)

        green_mask = 1-np.all(sample == [0, 255, 0], axis=-1).astype(int)
        green_mask = torch.from_numpy(green_mask.astype(np.float32)).unsqueeze(0)
        sample = torch.from_numpy(sample.astype(np.float32)).permute(2, 0, 1)/255
        sample = sample * green_mask

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
    landmarks = []
    path = os.path.join(landmark_path, str(imgname) + '.txt')
    with open(path, 'r') as f:
        for line in f:
            tmp = [np.float(i) for i in line.split(' ') if i != '\n']
            landmarks.append(tmp)
    landmarks = np.array(landmarks)/downscale  # 512 * 512

    Map_LE = list(np.hstack((range(17, 22), range(36, 42))))
    Map_RE = list(np.hstack((range(22, 27), range(42, 48))))
    Map_NO = list(range(29, 36))
    Map_MO = list(range(48, 68))
    # left eye
    Mean_LE = np.mean(landmarks[Map_LE], 0)
    L_LE = np.max((np.max(np.max(landmarks[Map_LE], 0) - np.min(landmarks[Map_LE], 0))/2, 16))
    Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
    # right eye
    Mean_RE = np.mean(landmarks[Map_RE], 0)
    L_RE = np.max((np.max(np.max(landmarks[Map_RE], 0) - np.min(landmarks[Map_RE], 0))/2, 16))
    Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
    # nose
    Mean_NO = np.mean(landmarks[Map_NO], 0)
    L_NO = np.max((np.max(np.max(landmarks[Map_NO], 0) - np.min(landmarks[Map_NO], 0))/2, 16))
    Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
    # mouth
    Mean_MO = np.mean(landmarks[Map_MO], 0)
    L_MO = np.max((np.max(np.max(landmarks[Map_MO], 0) - np.min(landmarks[Map_MO], 0))/2, 16))

    Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
    return Location_LE, Location_RE, Location_NO, Location_MO


class DS_lrhr(Dataset):
    def __init__(self, lr_path, hr_path, hr_size=256, scale=4, transform=None):
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
        # only get image if kernels are not used
        if cfg['datasets']['train']['apply_otf_downscale'] is False:
            lr_path = os.path.join(self.lr_path, os.path.basename(hr_path))
            lr_image = cv2.imread(lr_path)
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # checking for hr_size limitation
        if hr_image.shape[0] > self.hr_size or hr_image.shape[1] > self.hr_size:
            # image too big, random crop
            random_pos1 = random.randint(0, hr_image.shape[0]-self.hr_size)
            random_pos2 = random.randint(0, hr_image.shape[1]-self.hr_size)

            hr_image = hr_image[random_pos1:random_pos1+self.hr_size, random_pos2:random_pos2+self.hr_size]
            if cfg['datasets']['train']['apply_otf_downscale'] is False:
                lr_image = lr_image[int(random_pos1/self.scale):int((random_pos1/self.scale)+self.hr_size/self.scale),
                                    int(random_pos2/self.scale):int((random_pos2/self.scale)+self.hr_size/self.scale)]

        # OTFDownscale
        if cfg['datasets']['train']['apply_otf_downscale'] is True:
            filter_type = random.choices(
                cfg['datasets']['train']['otf_filter_types'],
                cum_weights=cfg['datasets']['train']['otf_filter_probs'])[0].upper()
            if filter_type == 'KERNEL':
                downscale_apply = transforms.ApplyKernel(
                    scale=cfg['scale'], kernels_path=cfg['datasets']['train']['kernel_path'],
                    kernel=None, pattern='kernelgan', kformat='npy', size=13,
                    permute=True, center=False)
            elif filter_type in ('NEAREST', 'BILINEAR', 'AREA', 'BICUBIC', 'LANCZOS'):
                downscale_apply = transforms.ApplyDownscale(
                    scale=cfg['scale'], filter_type=INTERP_MAP[filter_type])
            else:
                raise ValueError(f'{filter_type} is not a valid filter for OTF downscaling.')
            lr_image = downscale_apply(hr_image)

        # performing augmentation
        all_transforms = []
        # ColorJitter
        if cfg['datasets']['train']['ColorJitter'] is True:
            all_transforms.append(transforms.ColorJitter(
                p=augcfg['ColorJitter']['p'],
                brightness=augcfg['ColorJitter']['brightness'],
                contrast=augcfg['ColorJitter']['contrast'],
                saturation=augcfg['ColorJitter']['saturation'],
                hue=augcfg['ColorJitter']['hue']))
        # RandomGaussianNoise
        if cfg['datasets']['train']['RandomGaussianNoise'] is True:
            all_transforms.append(transforms.RandomGaussianNoise(
                p=augcfg['RandomGaussianNoise']['p'],
                mean=augcfg['RandomGaussianNoise']['mean'],
                var_limit=augcfg['RandomGaussianNoise']['var_limit'],
                prob_color=augcfg['RandomGaussianNoise']['prob_color'],
                multi=augcfg['RandomGaussianNoise']['multi'],
                mode=augcfg['RandomGaussianNoise']['mode'],
                sigma_calc=augcfg['RandomGaussianNoise']['sigma_calc']))
        # RandomPoissonNoise
        if cfg['datasets']['train']['RandomPoissonNoise'] is True:
            all_transforms.append(transforms.RandomPoissonNoise(
                p=augcfg['RandomPoissonNoise']['p'],
                prob_color=augcfg['RandomPoissonNoise']['prob_color'],
                scale_range=augcfg['RandomPoissonNoise']['scale_range']))
        # RandomSPNoise
        if cfg['datasets']['train']['RandomSPNoise'] is True:
            all_transforms.append(transforms.RandomSPNoise(
                p=augcfg['RandomSPNoise']['p'],
                prob=augcfg['RandomSPNoise']['prob']))
        # RandomSpeckleNoise
        if cfg['datasets']['train']['RandomSpeckleNoise'] is True:
            all_transforms.append(transforms.RandomSpeckleNoise(
                p=augcfg['RandomSpeckleNoise']['p'],
                mean=augcfg['RandomSpeckleNoise']['mean'],
                var_limit=augcfg['RandomSpeckleNoise']['var_limit'],
                prob_color=augcfg['RandomSpeckleNoise']['prob_color'],
                sigma_calc=augcfg['RandomSpeckleNoise']['sigma_calc']))
        # RandomCompression
        if cfg['datasets']['train']['RandomCompression'] is True:
            all_transforms.append(transforms.RandomCompression(
                p=augcfg['RandomCompression']['p'],
                min_quality=augcfg['RandomCompression']['min_quality'],
                max_quality=augcfg['RandomCompression']['max_quality'],
                compression_type=augcfg['RandomCompression']['compression_type']))
        # RandomAverageBlur
        if cfg['datasets']['train']['RandomAverageBlur'] is True:
            all_transforms.append(transforms.RandomAverageBlur(
                p=augcfg['RandomAverageBlur']['p'],
                kernel_size=augcfg['RandomAverageBlur']['kernel_size']))
        # RandomBilateralBlur
        if cfg['datasets']['train']['RandomBilateralBlur'] is True:
            all_transforms.append(transforms.RandomAverageBlur(
                p=augcfg['RandomBilateralBlur']['p'],
                kernel_size=augcfg['RandomBilateralBlur']['kernel_size'],
                sigmaX=augcfg['RandomBilateralBlur']['sigmaX'],
                sigmaY=augcfg['RandomBilateralBlur']['sigmaY']))
        # RandomBoxBlur
        if cfg['datasets']['train']['RandomBoxBlur'] is True:
            all_transforms.append(transforms.RandomBoxBlur(
                p=augcfg['RandomBoxBlur']['p'],
                kernel_size=augcfg['RandomBoxBlur']['kernel_size']))
        # RandomGaussianBlur
        if cfg['datasets']['train']['RandomGaussianBlur'] is True:
            all_transforms.append(transforms.RandomGaussianBlur(
                p=augcfg['RandomGaussianBlur']['p'],
                kernel_size=augcfg['RandomGaussianBlur']['kernel_size'],
                sigmaX=augcfg['RandomGaussianBlur']['sigmaX'],
                sigmaY=augcfg['RandomGaussianBlur']['sigmaY']))
        # RandomMedianBlur
        if cfg['datasets']['train']['RandomMedianBlur'] is True:
            all_transforms.append(transforms.RandomMedianBlur(
                p=augcfg['RandomMedianBlur']['p'],
                kernel_size=augcfg['RandomMedianBlur']['kernel_size']))
        # RandomMotionBlur
        if cfg['datasets']['train']['RandomMotionBlur'] is True:
            all_transforms.append(transforms.RandomMotionBlur(
                p=augcfg['RandomMotionBlur']['p'],
                kernel_size=augcfg['RandomMotionBlur']['kernel_size'],
                per_channel=augcfg['RandomMotionBlur']['per_channel']))
        # RandomComplexMotionBlur
        if cfg['datasets']['train']['RandomComplexMotionBlur'] is True:
            all_transforms.append(transforms.RandomComplexMotionBlur(
                p=augcfg['RandomComplexMotionBlur']['p'],
                size=augcfg['RandomComplexMotionBlur']['size'],
                complexity=augcfg['RandomComplexMotionBlur']['complexity'],
                eps=augcfg['RandomComplexMotionBlur']['eps']))
        # RandomAnIsoBlur
        if cfg['datasets']['train']['RandomAnIsoBlur'] is True:
            all_transforms.append(transforms.RandomAnIsoBlur(
                p=augcfg['RandomAnIsoBlur']['p'],
                min_kernel_size=augcfg['RandomAnIsoBlur']['min_kernel_size'],
                kernel_size=augcfg['RandomAnIsoBlur']['kernel_size'],
                sigmaX=augcfg['RandomAnIsoBlur']['sigmaX'],
                sigmaY=augcfg['RandomAnIsoBlur']['sigmaY'],
                angle=augcfg['RandomAnIsoBlur']['angle'],
                noise=augcfg['RandomAnIsoBlur']['noise'],
                scale=augcfg['RandomAnIsoBlur']['scale']))
        # RandomSincBlur
        if cfg['datasets']['train']['RandomSincBlur'] is True:
            all_transforms.append(transforms.RandomSincBlur(
                p=augcfg['RandomSincBlur']['p'],
                min_kernel_size=augcfg['RandomSincBlur']['min_kernel_size'],
                kernel_size=augcfg['RandomSincBlur']['kernel_size'],
                min_cutoff=augcfg['RandomSincBlur']['min_cutoff']))
        # BayerDitherNoise
        if cfg['datasets']['train']['BayerDitherNoise'] is True:
            all_transforms.append(transforms.BayerDitherNoise(
                p=augcfg['BayerDitherNoise']['p']))
        # FSDitherNoise
        if cfg['datasets']['train']['FSDitherNoise'] is True:
            all_transforms.append(transforms.FSDitherNoise(
                p=augcfg['FSDitherNoise']['p']))
        # FilterMaxRGB
        if cfg['datasets']['train']['FilterMaxRGB'] is True:
            all_transforms.append(transforms.FilterMaxRGB(
                p=augcfg['FilterMaxRGB']['p']))
        # FilterColorBalance
        if cfg['datasets']['train']['FilterColorBalance'] is True:
            all_transforms.append(transforms.FilterColorBalance(
                p=augcfg['FilterColorBalance']['p'],
                percent=augcfg['FilterColorBalance']['percent'],
                random_params=augcfg['FilterColorBalance']['random_params']))
        # FilterUnsharp
        if cfg['datasets']['train']['FilterUnsharp'] is True:
            all_transforms.append(transforms.FilterUnsharp(
                p=augcfg['FilterUnsharp']['p'],
                blur_algo=augcfg['FilterUnsharp']['blur_algo'],
                kernel_size=augcfg['FilterUnsharp']['kernel_size'],
                strength=augcfg['FilterUnsharp']['strength'],
                unsharp_algo=augcfg['FilterUnsharp']['unsharp_algo']))
        # FilterCanny
        if cfg['datasets']['train']['FilterCanny'] is True:
            all_transforms.append(transforms.FilterCanny(
                p=augcfg['FilterCanny']['p'],
                sigma=augcfg['FilterCanny']['sigma'],
                bin_thresh=augcfg['FilterCanny']['bin_thresh'],
                threshold=augcfg['FilterCanny']['threshold']))
        # SimpleQuantize
        if cfg['datasets']['train']['SimpleQuantize'] is True:
            all_transforms.append(transforms.SimpleQuantize(
                p=augcfg['SimpleQuantize']['p'],
                rgb_range=augcfg['SimpleQuantize']['rgb_range']))
        # KMeansQuantize
        if cfg['datasets']['train']['KMeansQuantize'] is True:
            all_transforms.append(transforms.KMeansQuantize(
                p=augcfg['KMeansQuantize']['p'],
                n_colors=augcfg['KMeansQuantize']['n_colors']))
        # CLAHE
        if cfg['datasets']['train']['CLAHE'] is True:
            all_transforms.append(transforms.CLAHE(
                p=augcfg['CLAHE']['p'],
                clip_limit=augcfg['CLAHE']['clip_limit'],
                tile_grid_size=augcfg['CLAHE']['tile_grid_size']))
        # RandomGamma
        if cfg['datasets']['train']['RandomGamma'] is True:
            all_transforms.append(transforms.RandomGamma(
                p=augcfg['RandomGamma']['p'],
                gamma_range=augcfg['RandomGamma']['gamma_range'],
                gain=augcfg['RandomGamma']['gain']))
        # Superpixels
        if cfg['datasets']['train']['Superpixels'] is True:
            all_transforms.append(transforms.Superpixels(
                p=augcfg['Superpixels']['p'],
                p_replace=augcfg['Superpixels']['p_replace'],
                n_segments=augcfg['Superpixels']['n_segments'],
                cs=augcfg['Superpixels']['cs'],
                algo=augcfg['Superpixels']['algo'],
                n_iters=augcfg['Superpixels']['n_iters'],
                kind=augcfg['Superpixels']['kind'],
                reduction=augcfg['Superpixels']['reduction'],
                max_size=augcfg['Superpixels']['max_size'],
                interpolation=augcfg['Superpixels']['interpolation']))
        # RandomCameraNoise
        if cfg['datasets']['train']['RandomCameraNoise'] is True:
            all_transforms.append(transforms.RandomCameraNoise(
                p=augcfg['RandomCameraNoise']['p'],
                demosaic_fn=augcfg['RandomCameraNoise']['demosaic_fn'],
                xyz_arr=augcfg['RandomCameraNoise']['xyz_arr'],
                rg_range=augcfg['RandomCameraNoise']['rg_range'],
                bg_range=augcfg['RandomCameraNoise']['bg_range'],
                random_params=augcfg['RandomCameraNoise']['random_params']))

        # BW augmentations
        """
        # [APPLY] BayerBWDitherNoise / 1ch output
        all_transforms.append(transforms.BayerBWDitherNoise(p=0.5))
        # [APPLY] BinBWDitherNoise / 1ch output
        all_transforms.append(transforms.BinBWDitherNoise(p=0.5))
        # FSBWDitherNoise / 1ch output
        all_transforms.append(transforms.FSBWDitherNoise(p=0.5, samplingF = 1))
        # [APPLY] RandomBWDitherNoise / 1ch output
        all_transforms.append(transforms.RandomBWDitherNoise(p=0.5))
        """
        # [BROKEN] RandomChromaticAberration
        # all_transforms.append(transforms.RandomChromaticAberration(p=0.5, radial_blur=True,
        # strength=1.0, jitter=0, alpha=0.0,
        # random_params=False))

        # randomly shuffle transforms
        random.shuffle(all_transforms)
        # apply transforms
        transform = transforms.Compose(all_transforms)
        lr_image = transform(lr_image)

        # to tensor
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1)/255
        lr_image = torch.from_numpy(lr_image).permute(2, 0, 1)/255

        # if generator is DFDNet, change image range to [-1,1] and also pass landmarks
        if cfg['network_G']['netG'] == 'DFDNet':
            hr_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(hr_image)
            lr_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(lr_image)

            landmarks = get_part_location(
                landmark_path=cfg['network_G']['landmarkpath'],
                imgname=os.path.basename(hr_path), downscale=1)
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

            landmarks = get_part_location(
                landmark_path=cfg['network_G']['val_landmarkpath'],
                imgname=os.path.basename(hr_path), downscale=1)
            return lr_image, hr_image, lr_path, landmarks
        else:
            return lr_image, hr_image, lr_path


class DS_inpaint_TF(Dataset):
    def __init__(self):
        tfrecord_path = cfg['datasets']['train']['tfrecord_path']
        self.mask_dir = cfg['datasets']['train']['masks']
        self.mask_files = glob.glob(self.mask_dir + '/**/*.png', recursive=True)

        self.HR_size = cfg['datasets']['train']['HR_size']
        # self.batch_size = cfg['datasets']['train']['batch_size']

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

        # resize
        # sample = cv2.resize(sample, (self.HR_size, self.HR_size), interpolation=cv2.INTER_AREA)

        # random crop
        # checking for hr_size limitation
        if sample.shape[0] > self.HR_size or sample.shape[1] > self.HR_size:
            # image too big, random crop
            random_pos1 = random.randint(0, sample.shape[0]-self.HR_size)
            random_pos2 = random.randint(0, sample.shape[1]-self.HR_size)
            sample = sample[random_pos1:random_pos1+self.HR_size, random_pos2:random_pos2+self.HR_size]

        # if edges are required
        if (cfg['network_G']['netG'] == 'EdgeConnect'
                or cfg['network_G']['netG'] == 'PRVS'
                or cfg['network_G']['netG'] == 'CTSDG'):
            grayscale = cv2.cvtColor(np.array(sample), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(grayscale, 100, 150)
            grayscale = torch.from_numpy(grayscale).unsqueeze(0)/255
            edges = torch.from_numpy(edges).unsqueeze(0).type(torch.float)

        if random.uniform(0, 1) < 0.5:
            # generating mask automatically with 50% chance
            mask = random_mask(height=self.HR_size, width=self.HR_size)
            mask = torch.from_numpy(mask)

        else:
            # load random mask from folder
            mask = cv2.imread(random.choice([x for x in self.mask_files]), cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask, (self.HR_size, self.HR_size), interpolation=cv2.INTER_NEAREST)
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

        png = svg2png(bytestring=np.fromstring(
            np.array(data['data']), np.uint8).tobytes(), dpi=dpi,
                      output_width=output_width, output_height=output_height)
        
        # background fix
        hr_image = Image.open(BytesIO(png)).convert('RGBA')
        background = Image.new('RGBA', hr_image.size, (255, 255, 255))
        hr_image = Image.alpha_composite(background, hr_image)
        hr_image = hr_image.convert('RGB')

        # resize
        lr_image = hr_image.resize((64, 64), Image.ANTIALIAS)

        # to tensor
        lr_image = transforms.ToTensor()(lr_image)
        hr_image = transforms.ToTensor()(hr_image)
        return 0, lr_image, hr_image
