"""
atrousinpainter.py (21-12-20)
https://github.com/junqingchang/inpainting-with-atrous-convolutions/blob/6215820b4769083e1385655c1513b999d34cac5f/atrousinpainter.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AtrousInpainter(nn.Module):
    def __init__(self):
        super(AtrousInpainter, self).__init__()
        self.feature_extract = FeatureExtraction()
        self.aspp = ASPP(in_channels=1024, out_channels=512)
        self.latent = nn.Conv2d(in_channels = 512, out_channels = 1024,
                          kernel_size = 1, stride=1, padding=0)

        self.generator = ImageGenerator()

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.aspp(x)
        x = self.latent(x)
        x = self.generator(x)
        return x


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        return x



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size = 1,
                            padding = 0,
                            dilation=1,
                            bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size = 3,
                            stride=1,
                            padding = 6,
                            dilation = 6,
                            bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size = 3,
                            stride=1,
                            padding = 12,
                            dilation = 12,
                            bias=False)

        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size = 3,
                            stride=1,
                            padding = 18,
                            dilation = 18,
                            bias=False)

        self.bn4 = nn.BatchNorm2d(out_channels)

        self.conv5 = nn.Conv2d(in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size = 1,
                            stride=1,
                            padding = 0,
                            dilation=1,
                            bias=False)

        self.bn5 = nn.BatchNorm2d(out_channels)

        self.convf = nn.Conv2d(in_channels = out_channels * 5,
                            out_channels = out_channels,
                            kernel_size = 1,
                            stride=1,
                            padding = 0,
                            dilation=1,
                            bias=False)

        self.bnf = nn.BatchNorm2d(out_channels)

        self.adapool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)

        x5 = self.adapool(x)
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size = tuple(x4.shape[-2:]), mode='bilinear', align_corners=True)

        x = torch.cat((x1,x2,x3,x4,x5), dim = 1)
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)
        return x


class ImageGenerator(nn.Module):
    def __init__(self):
        super(ImageGenerator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.ConvTranspose2d(64, 3, kernel_size=2)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.out(x)

        return x

"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.classifier = nn.Linear(1024*8*8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = x.view((-1, 1024*8*8))

        x = self.classifier(x)
        x = self.sigmoid(x)
        return x
"""
