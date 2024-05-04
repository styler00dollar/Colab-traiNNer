import yaml
import os
import cv2
import torch
from torch.utils.data import Dataset

INTERP_MAP = {
    "NEAREST": cv2.INTER_NEAREST,
    "BILINEAR": cv2.INTER_LINEAR,
    "AREA": cv2.INTER_AREA,
    "BICUBIC": cv2.INTER_CUBIC,
    "LANCZOS": cv2.INTER_LANCZOS4,
}

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
with open("aug_config.yaml", "r") as ymlfile:
    augcfg = yaml.safe_load(ymlfile)

if (
    cfg["datasets"]["train"]["mode"] == "DS_inpaint_TF"
    or cfg["datasets"]["train"]["mode"] == "DS_svg_TF"
):
    pass

if cfg["datasets"]["train"]["mode"] == "DS_svg_TF":
    pass

if cfg["datasets"]["train"]["loading_backend"] == "PIL":
    pass


class DS_lrhr(Dataset):
    def __init__(self, lr_path, hr_path, hr_size=256, scale=4, transform=None):
        self.samples = []
        for hr_path, _, fnames in sorted(os.walk(hr_path)):
            for fname in sorted(fnames):
                path = os.path.join(hr_path, fname)
                if ".pt" in path or ".jpg" in path or ".webp" in path:
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

        with torch.inference_mode():
            hr_image = torch.load(hr_path, map_location="cpu")
            lr_path = os.path.join(self.lr_path, os.path.basename(hr_path))
            lr_image = torch.load(lr_path, map_location="cpu")

            return 0, lr_image.squeeze(0), hr_image.squeeze(0)


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
        return 0, 0, 0
