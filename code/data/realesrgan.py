"""
15-3-22
https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/data/realesrgan_dataset.py
https://github.com/xinntao/Real-ESRGAN/blob/35ee6f781e9a5a80d5f2f1efb9102c9899a81ae1/realesrgan/models/realesrgan_model.py
"""
import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import (
    circular_lowpass_kernel,
    random_mixed_kernels,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import (
    FileClient,
    get_root_logger,
    imfrombytes,
    img2tensor,
    DiffJPEG,
    USMSharp,
)
from basicsr.utils.img_process_util import filter2D
import yaml
import pytorch_lightning as pl
import torch.nn.functional as F


class RealESRGANDataset(pl.LightningDataModule):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, hr_path, hr_size=256, scale=4):
        super(RealESRGANDataset, self).__init__()

        self.samples = []
        for hr_path, _, fnames in sorted(os.walk(hr_path)):
            for fname in sorted(fnames):
                path = os.path.join(hr_path, fname)
                if ".png" in path or ".jpg" in path or ".webp" in path:
                    self.samples.append(path)

        self.hr_size = hr_size
        self.scale = scale

        with open("realesrgan_aug_config.yaml", "r") as ymlfile:
            opt = yaml.safe_load(ymlfile)

        with open("config.yaml", "r") as ymlfile:
            self.config = yaml.safe_load(ymlfile)

        if self.config["datasets"]["train"]["loading_backend"] == "turboJPEG":
            from turbojpeg import TurboJPEG

            self.jpeg_reader = TurboJPEG()

        # blur settings for the first degradation
        self.blur_kernel_size = opt["blur_kernel_size"]
        self.kernel_list = opt["kernel_list"]
        self.kernel_prob = opt["kernel_prob"]  # a list for each kernel probability
        self.blur_sigma = opt["blur_sigma"]
        self.betag_range = opt[
            "betag_range"
        ]  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt["betap_range"]  # betap used in plateau blur kernels
        self.sinc_prob = opt["sinc_prob"]  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt["blur_kernel_size2"]
        self.kernel_list2 = opt["kernel_list2"]
        self.kernel_prob2 = opt["kernel_prob2"]
        self.blur_sigma2 = opt["blur_sigma2"]
        self.betag_range2 = opt["betag_range2"]
        self.betap_range2 = opt["betap_range2"]
        self.sinc_prob2 = opt["sinc_prob2"]

        # a final sinc filter
        self.final_sinc_prob = opt["final_sinc_prob"]

        self.opt = opt

        self.kernel_range = [
            2 * v + 1 for v in range(3, 11)
        ]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(
            21, 21
        ).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_gt = self.samples[index]

        if self.config["datasets"]["train"]["loading_backend"] == "OpenCV":
            img_gt = cv2.imread(img_gt)
        elif self.config["datasets"]["train"]["loading_backend"] == "turboJPEG":
            img_gt = self.jpeg_reader.decode(
                open(img_gt, "rb").read(), 1
            )  # 0 = rgb, 1 = bgr

        img_gt = img_gt.astype(np.float32) / 255.0

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt["use_hflip"], self.opt["use_rot"])

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = 400
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(
                img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top : top + crop_pad_size, left : left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt["sinc_prob"]:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt["sinc_prob2"]:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt["final_sinc_prob"]:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel1 = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        # you need to return tensors because of lightning
        return img_gt, kernel1, kernel2, sinc_kernel

    def __len__(self):
        return len(self.samples)


# Due to CUDA reasons, a workaround like in the original RealESRGAN repo needs to be integrated,
# so multithreading can be used in the dataloader


class RealESRGANDatasetApply(pl.LightningDataModule):
    def __init__(self, device):
        super(RealESRGANDatasetApply, self).__init__()

        with open("realesrgan_aug_config.yaml", "r") as ymlfile:
            self.opt = yaml.safe_load(ymlfile)

        with open("config.yaml", "r") as ymlfile:
            self.config = yaml.safe_load(ymlfile)

        # the .to statements need to be inside the loop, because __init__ is on cpu
        # and due to multi-gpu the device can change in forward()
        self.jpeger = DiffJPEG(
            differentiable=False
        )  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp()  # do usm sharpening

    def forward(self, img_gt, kernel1, kernel2, sinc_kernel, device):
        self.jpeger = self.jpeger.to(device)
        self.usm_sharpener = self.usm_sharpener.to(device)

        self.gt = img_gt.to(device)
        ori_h, ori_w = self.gt.size()[2:4]

        self.gt_usm = self.usm_sharpener(self.gt)
        self.kernel1 = kernel1.to(device)
        self.kernel2 = kernel2.to(device)
        self.sinc_kernel = sinc_kernel.to(device)

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(self.gt_usm, self.kernel1)
        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.opt["resize_prob"])[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.opt["resize_range"][1])
        elif updown_type == "down":
            scale = np.random.uniform(self.opt["resize_range"][0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.opt["gray_noise_prob"]
        if np.random.uniform() < self.opt["gaussian_noise_prob"]:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.opt["noise_range"],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt["poisson_scale_range"],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
            )
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range"])
        out = torch.clamp(
            out, 0, 1
        )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.opt["second_blur_prob"]:
            out = filter2D(out, self.kernel2)
        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.opt["resize_prob2"])[
            0
        ]
        if updown_type == "up":
            scale = np.random.uniform(1, self.opt["resize_range2"][1])
        elif updown_type == "down":
            scale = np.random.uniform(self.opt["resize_range2"][0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.opt["gray_noise_prob2"]
        if np.random.uniform() < self.opt["gaussian_noise_prob2"]:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.opt["noise_range2"],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt["poisson_scale_range2"],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
            )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(
                out,
                size=(ori_h // self.config["scale"], ori_w // self.config["scale"]),
                mode=mode,
            )
            out = filter2D(out, self.sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(
                out,
                size=(ori_h // self.config["scale"], ori_w // self.config["scale"]),
                mode=mode,
            )
            out = filter2D(out, self.sinc_kernel)

        # clamp and round
        self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        # random crop
        gt_size = self.config["datasets"]["train"]["HR_size"]
        (self.gt, self.gt_usm), self.lq = paired_random_crop(
            [self.gt, self.gt_usm], self.lq, gt_size, self.config["scale"]
        )

        self.lq = (
            self.lq.contiguous()
        )  # for the warning: grad and param do not obey the gradient layout contract

        return self.lq.squeeze(0), self.gt_usm.squeeze(0), self.gt.squeeze(0)
