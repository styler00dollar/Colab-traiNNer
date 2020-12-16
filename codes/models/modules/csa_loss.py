"""
# needs to be merged with loss.py
loss.py (16-12-20)
https://github.com/Yukariin/CSA_pytorch/blob/master/loss.py
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms


def denorm(x):
    out = (x + 1) / 2 # [-1,1] -> [0,1]
    return out.clamp_(0, 1)


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        self.enc_4 = nn.Sequential(*vgg16.features[17:23])

        #print(self.enc_1)
        #print(self.enc_2)
        #print(self.enc_3)
        #print(self.enc_4)

        # fix the encoder
        for i in range(4):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.vgg = VGG16FeatureExtractor()
        self.l2 = nn.MSELoss()

    def forward(self, csa, csa_d, target, mask):
        # https://pytorch.org/docs/stable/torchvision/models.html
        # Pre-trained VGG16 model expect input images normalized in the same way.
        # The images have to be loaded in to a range of [0, 1]
        # and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        t = denorm(target) # [-1,1] -> [0,1]
        t = self.normalize(t[0]) # BxCxHxW -> CxHxW -> normalize
        t = t.unsqueeze(0) # CxHxW -> BxCxHxW

        vgg_gt = self.vgg(t)
        vgg_gt = vgg_gt[-1]

        mask_r = F.interpolate(mask, size=csa.size()[2:])

        lossvalue = self.l2(csa*mask_r, vgg_gt*mask_r) + self.l2(csa_d*mask_r, vgg_gt*mask_r)
        return lossvalue
