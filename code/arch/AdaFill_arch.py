"""
network.py (3-2-20)
https://github.com/ChajinShin/AdaFill-Image_Inpainting/blob/main/Model/AdaFill/src/network.py
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


class ResModule(pl.LightningModule):
    def __init__(self, num_features, normalization):
        super(ResModule, self).__init__()
        self.block = nn.Sequential(
            normalization(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            normalization(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class InpaintNet(pl.LightningModule):
    def __init__(self):
        super(InpaintNet, self).__init__()
        cnum = 64
        num_of_resblock = 8
        normalization_config = "batch_norm"

        if normalization_config == "batch_norm":
            normalization = nn.BatchNorm2d
        elif normalization_config == "instance_norm":
            normalization = nn.InstanceNorm2d
        else:
            raise ValueError(
                "batch normalization or instance normalization is only available"
            )

        self.enc_conv1 = nn.Conv2d(
            in_channels=4, out_channels=cnum, kernel_size=3, stride=1, padding=1
        )  # cnum, 128, 128
        self.enc_norm1 = normalization(num_features=cnum)
        self.enc_activation1 = nn.ReLU(inplace=True)
        self.enc_conv2 = nn.Conv2d(
            in_channels=cnum, out_channels=2 * cnum, kernel_size=3, stride=2, padding=1
        )  # 2cnum, 64, 64
        self.enc_norm2 = normalization(num_features=2 * cnum)
        self.enc_activation2 = nn.ReLU(inplace=True)
        self.enc_conv3 = nn.Conv2d(
            in_channels=2 * cnum,
            out_channels=4 * cnum,
            kernel_size=3,
            stride=2,
            padding=1,
        )  # 4cnum, 32, 32

        res = [ResModule(4 * cnum, normalization) for _ in range(num_of_resblock)]
        self.res_module = nn.Sequential(*res)

        self.dec_norm1 = normalization(4 * cnum)
        self.dec_activation1 = nn.ReLU(inplace=True)
        self.dec_conv1 = nn.Conv2d(
            in_channels=4 * cnum,
            out_channels=2 * cnum,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # 2*cnum, 64, 64
        self.dec_norm2 = normalization(num_features=2 * cnum)
        self.dec_activation2 = nn.ReLU(inplace=True)
        self.dec_conv2 = nn.Conv2d(
            in_channels=2 * cnum, out_channels=cnum, kernel_size=3, stride=1, padding=1
        )  # cnum, 128, 128
        self.dec_norm3 = normalization(num_features=cnum)
        self.dec_activation3 = nn.ReLU(inplace=True)
        self.dec_conv3 = nn.Conv2d(
            in_channels=cnum, out_channels=3, kernel_size=3, stride=1, padding=1
        )  # 3, 128, 128
        self.tanh = nn.Tanh()

    def forward(self, image, mask):
        x = torch.cat((image, mask), 1)
        # --------- encoder ----------------
        x = self.enc_conv1(x)
        x = self.enc_norm1(x)
        x = self.enc_activation1(x)
        size_1x = [x.size(2), x.size(3)]

        x = self.enc_conv2(x)
        x = self.enc_norm2(x)
        x = self.enc_activation2(x)
        size_2x = [x.size(2), x.size(3)]

        x = self.enc_conv3(x)

        # --------- res module ----------------
        x = self.res_module(x)

        # --------- decoder ----------------
        x = self.dec_norm1(x)
        x = self.dec_activation1(x)
        x = nn.functional.interpolate(x, size=size_2x)
        x = self.dec_conv1(x)

        x = self.dec_norm2(x)
        x = self.dec_activation2(x)
        x = nn.functional.interpolate(x, size=size_1x)
        x = self.dec_conv2(x)

        x = self.dec_norm3(x)
        x = self.dec_activation3(x)
        x = self.dec_conv3(x)
        x = self.tanh(x)
        return x
