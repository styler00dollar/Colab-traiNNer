"""
LBAMModel.py (18-12-20)
https://github.com/Vious/LBAM_Pytorch/blob/master/models/LBAMModel.py

forwardAttentionLayer.py (18-12-20)
https://github.com/Vious/LBAM_Pytorch/blob/98c2ae70486f4ba3ab86d4345e586e7841cfe343/models/forwardAttentionLayer.py

reverseAttentionLayer.py (18-12-20)
https://github.com/Vious/LBAM_Pytorch/blob/98c2ae70486f4ba3ab86d4345e586e7841cfe343/models/reverseAttentionLayer.py

ActivationFunction.py (18-12-20)
https://github.com/Vious/LBAM_Pytorch/blob/98c2ae70486f4ba3ab86d4345e586e7841cfe343/models/ActivationFunction.py
"""

import math
import torch
from torch.nn.parameter import Parameter
from torch import nn
from torchvision import models
import pytorch_lightning as pl

# asymmetric gaussian shaped activation function g_A
class GaussActivation(pl.LightningModule):
    def __init__(self, a, mu, sigma1, sigma2):
        super(GaussActivation, self).__init__()

        self.a = Parameter(torch.tensor(a, dtype=torch.float32))
        self.mu = Parameter(torch.tensor(mu, dtype=torch.float32))
        self.sigma1 = Parameter(torch.tensor(sigma1, dtype=torch.float32))
        self.sigma2 = Parameter(torch.tensor(sigma2, dtype=torch.float32))


    def forward(self, inputFeatures):

        self.a.data = torch.clamp(self.a.data, 1.01, 6.0)
        self.mu.data = torch.clamp(self.mu.data, 0.1, 3.0)
        self.sigma1.data = torch.clamp(self.sigma1.data, 0.5, 2.0)
        self.sigma2.data = torch.clamp(self.sigma2.data, 0.5, 2.0)

        lowerThanMu = inputFeatures < self.mu
        largerThanMu = inputFeatures >= self.mu

        leftValuesActiv = self.a * torch.exp(- self.sigma1 * ( (inputFeatures - self.mu) ** 2 ) )
        leftValuesActiv.masked_fill_(largerThanMu, 0.0)

        rightValueActiv = 1 + (self.a - 1) * torch.exp(- self.sigma2 * ( (inputFeatures - self.mu) ** 2 ) )
        rightValueActiv.masked_fill_(lowerThanMu, 0.0)

        output = leftValuesActiv + rightValueActiv

        return output

# mask updating functions, we recommand using alpha that is larger than 0 and lower than 1.0
class MaskUpdate(pl.LightningModule):
    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()

        self.updateFunc = nn.ReLU(True)
        #self.alpha = Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.alpha = alpha
    def forward(self, inputMaskMap):
        """ self.alpha.data = torch.clamp(self.alpha.data, 0.6, 0.8)
        print(self.alpha) """

        return torch.pow(self.updateFunc(inputMaskMap), self.alpha)


import math
import torch
from torch import nn
#from models.ActivationFunction import GaussActivation, MaskUpdate
#from models.weightInitial import weights_init


# learnable reverse attention conv
class ReverseMaskConv(pl.LightningModule):
    def __init__(self, inputChannels, outputChannels, kernelSize=4, stride=2,
        padding=1, dilation=1, groups=1, convBias=False):
        super(ReverseMaskConv, self).__init__()

        self.reverseMaskConv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, \
            dilation, groups, bias=convBias)

        #self.reverseMaskConv.apply(weights_init())

        self.activationFuncG_A = GaussActivation(1.1, 1.0, 0.5, 0.5)
        self.updateMask = MaskUpdate(0.8)

    def forward(self, inputMasks):
        maskFeatures = self.reverseMaskConv(inputMasks)

        maskActiv = self.activationFuncG_A(maskFeatures)

        maskUpdate = self.updateMask(maskFeatures)

        return maskActiv, maskUpdate

# learnable reverse attention layer, including features activation and batchnorm
class ReverseAttention(pl.LightningModule):
    def __init__(self, inputChannels, outputChannels, bn=False, activ='leaky', \
        kernelSize=4, stride=2, padding=1, outPadding=0,dilation=1, groups=1,convBias=False, bnChannels=512):
        super(ReverseAttention, self).__init__()

        self.conv = nn.ConvTranspose2d(inputChannels, outputChannels, kernel_size=kernelSize, \
            stride=stride, padding=padding, output_padding=outPadding, dilation=dilation, groups=groups,bias=convBias)

        #self.conv.apply(weights_init())

        if bn:
            self.bn = nn.BatchNorm2d(bnChannels)

        if activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2, False)
        elif activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        else:
            pass

    def forward(self, ecFeaturesSkip, dcFeatures, maskFeaturesForAttention):
        nextDcFeatures = self.conv(dcFeatures)

        # note that encoder features are ahead, it's important tor make forward attention map ahead
        # of reverse attention map when concatenate, we do it in the LBAM model forward function
        concatFeatures = torch.cat((ecFeaturesSkip, nextDcFeatures), 1)

        outputFeatures = concatFeatures * maskFeaturesForAttention

        if hasattr(self, 'bn'):
            outputFeatures = self.bn(outputFeatures)
        if hasattr(self, 'activ'):
            outputFeatures = self.activ(outputFeatures)

        return outputFeatures


import math
import torch
from torch import nn
#from models.ActivationFunction import GaussActivation, MaskUpdate
#from models.weightInitial import weights_init

# learnable forward attention conv layer
class ForwardAttentionLayer(pl.LightningModule):
    def __init__(self, inputChannels, outputChannels, kernelSize, stride,
        padding, dilation=1, groups=1, bias=False):
        super(ForwardAttentionLayer, self).__init__()

        self.conv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, dilation, \
            groups, bias)

        if inputChannels == 4:
            self.maskConv = nn.Conv2d(3, outputChannels, kernelSize, stride, padding, dilation, \
                groups, bias)
        else:
            self.maskConv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, \
                dilation, groups, bias)

        #self.conv.apply(weights_init())
        #self.maskConv.apply(weights_init())

        self.activationFuncG_A = GaussActivation(1.1, 2.0, 1.0, 1.0)
        self.updateMask = MaskUpdate(0.8)

    def forward(self, inputFeatures, inputMasks):
        convFeatures = self.conv(inputFeatures)
        maskFeatures = self.maskConv(inputMasks)
        #convFeatures_skip = convFeatures.clone()

        maskActiv = self.activationFuncG_A(maskFeatures)
        convOut = convFeatures * maskActiv

        maskUpdate = self.updateMask(maskFeatures)

        return convOut, maskUpdate, convFeatures, maskActiv

# forward attention gather feature activation and batchnorm
class ForwardAttention(pl.LightningModule):
    def __init__(self, inputChannels, outputChannels, bn=False, sample='down-4', \
        activ='leaky', convBias=False):
        super(ForwardAttention, self).__init__()

        if sample == 'down-4':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 4, 2, 1, bias=convBias)
        elif sample == 'down-5':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 5, 2, 2, bias=convBias)
        elif sample == 'down-7':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 7, 2, 3, bias=convBias)
        elif sample == 'down-3':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 3, 2, 1, bias=convBias)
        else:
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 3, 1, 1, bias=convBias)

        if bn:
            self.bn = nn.BatchNorm2d(outputChannels)

        if activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2, False)
        elif activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        else:
            pass

    def forward(self, inputFeatures, inputMasks):
        features, maskUpdated, convPreF, maskActiv = self.conv(inputFeatures, inputMasks)

        if hasattr(self, 'bn'):
            features = self.bn(features)
        if hasattr(self, 'activ'):
            features = self.activ(features)

        return features, maskUpdated, convPreF, maskActiv


import torch
import torch.nn as nn
from torchvision import models
#from models.forwardAttentionLayer import ForwardAttention
#from models.reverseAttentionLayer import ReverseAttention, ReverseMaskConv
#from models.weightInitial import weights_init

#VGG16 feature extract
class VGG16FeatureExtractor(pl.LightningModule):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('./vgg16-397923af.pth'))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class LBAMModel(pl.LightningModule):
    def __init__(self, inputChannels=4, outputChannels=3):
        super(LBAMModel, self).__init__()

        # default kernel is of size 4X4, stride 2, padding 1,
        # and the use of biases are set false in default ReverseAttention class.
        self.ec1 = ForwardAttention(inputChannels, 64, bn=False)
        self.ec2 = ForwardAttention(64, 128)
        self.ec3 = ForwardAttention(128, 256)
        self.ec4 = ForwardAttention(256, 512)

        for i in range(5, 8):
            name = 'ec{:d}'.format(i)
            setattr(self, name, ForwardAttention(512, 512))

        # reverse mask conv
        self.reverseConv1 = ReverseMaskConv(3, 64)
        self.reverseConv2 = ReverseMaskConv(64, 128)
        self.reverseConv3 = ReverseMaskConv(128, 256)
        self.reverseConv4 = ReverseMaskConv(256, 512)
        self.reverseConv5 = ReverseMaskConv(512, 512)
        self.reverseConv6 = ReverseMaskConv(512, 512)

        self.dc1 = ReverseAttention(512, 512, bnChannels=1024)
        self.dc2 = ReverseAttention(512 * 2, 512, bnChannels=1024)
        self.dc3 = ReverseAttention(512 * 2, 512, bnChannels=1024)
        self.dc4 = ReverseAttention(512 * 2, 256, bnChannels=512)
        self.dc5 = ReverseAttention(256 * 2, 128, bnChannels=256)
        self.dc6 = ReverseAttention(128 * 2, 64, bnChannels=128)
        self.dc7 = nn.ConvTranspose2d(64 * 2, outputChannels, kernel_size=4, stride=2, padding=1, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, inputImgs, masks):
        inputImgs = torch.cat((inputImgs, masks), 1).type(torch.cuda.FloatTensor)
        masks = torch.cat([masks,masks,masks],1).type(torch.cuda.FloatTensor)

        ef1, mu1, skipConnect1, forwardMap1 = self.ec1(inputImgs, masks)
        ef2, mu2, skipConnect2, forwardMap2 = self.ec2(ef1, mu1)
        ef3, mu3, skipConnect3, forwardMap3 = self.ec3(ef2, mu2)
        ef4, mu4, skipConnect4, forwardMap4 = self.ec4(ef3, mu3)
        ef5, mu5, skipConnect5, forwardMap5 = self.ec5(ef4, mu4)
        ef6, mu6, skipConnect6, forwardMap6 = self.ec6(ef5, mu5)
        ef7, _, _, _ = self.ec7(ef6, mu6)


        reverseMap1, revMu1 = self.reverseConv1(1 - masks)
        reverseMap2, revMu2 = self.reverseConv2(revMu1)
        reverseMap3, revMu3 = self.reverseConv3(revMu2)
        reverseMap4, revMu4 = self.reverseConv4(revMu3)
        reverseMap5, revMu5 = self.reverseConv5(revMu4)
        reverseMap6, _ = self.reverseConv6(revMu5)

        concatMap6 = torch.cat((forwardMap6, reverseMap6), 1)
        dcFeatures1 = self.dc1(skipConnect6, ef7, concatMap6)

        concatMap5 = torch.cat((forwardMap5, reverseMap5), 1)
        dcFeatures2 = self.dc2(skipConnect5, dcFeatures1, concatMap5)

        concatMap4 = torch.cat((forwardMap4, reverseMap4), 1)
        dcFeatures3 = self.dc3(skipConnect4, dcFeatures2, concatMap4)

        concatMap3 = torch.cat((forwardMap3, reverseMap3), 1)
        dcFeatures4 = self.dc4(skipConnect3, dcFeatures3, concatMap3)

        concatMap2 = torch.cat((forwardMap2, reverseMap2), 1)
        dcFeatures5 = self.dc5(skipConnect2, dcFeatures4, concatMap2)

        concatMap1 = torch.cat((forwardMap1, reverseMap1), 1)
        dcFeatures6 = self.dc6(skipConnect1, dcFeatures5, concatMap1)

        dcFeatures7 = self.dc7(dcFeatures6)

        output = (self.tanh(dcFeatures7) + 1) / 2

        return output
