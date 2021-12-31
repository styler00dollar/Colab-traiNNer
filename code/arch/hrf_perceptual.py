"""
28-Okt-21
https://github.com/saic-mdal/lama/blob/6bb704738d4e791106d8e87099d80831999901fc/saicinpainting/training/losses/perceptual.py
https://github.com/saic-mdal/lama/blob/6bb704738d4e791106d8e87099d80831999901fc/models/ade20k/base.py
https://github.com/saic-mdal/lama/blob/6bb704738d4e791106d8e87099d80831999901fc/saicinpainting/utils.py
https://github.com/saic-mdal/lama/blob/6bb704738d4e791106d8e87099d80831999901fc/models/ade20k/resnet.py
https://github.com/saic-mdal/lama/blob/6bb704738d4e791106d8e87099d80831999901fc/models/ade20k/utils.py
"""
"""Modified from https://github.com/CSAILVision/semantic-segmentation-pytorch"""
from scipy.io import loadmat
from torch.nn.modules import BatchNorm2d
from torch.nn import BatchNorm2d
import math
import numpy as np
import os
import os.path
import pandas as pd
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

NUM_CLASS = 150
"""
base_path = os.path.dirname(os.path.abspath(__file__))  # current file path
colors_path = os.path.join(base_path, 'color150.mat')
classes_path = os.path.join(base_path, 'object150_info.csv')

segm_options = dict(colors=loadmat(colors_path)['colors'],
                    classes=pd.read_csv(classes_path),)
"""
model_urls = {
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
}


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def color_encode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet18']))
    return model

    
def check_and_warn_input_range(tensor, min_value, max_value, name):
    actual_min = tensor.min()
    actual_max = tensor.max()
    if actual_min < min_value or actual_max > max_value:
        warnings.warn(f"{name} must be in {min_value}..{max_value} range, but it ranges {actual_min}..{actual_max}")


class NormalizeTensor:
    def __init__(self, mean, std, inplace=False):
        """Normalize a tensor image with mean and standard deviation.
        .. note::
            This transform acts out of place by default, i.e., it does not mutates the input tensor.
        See :class:`~torchvision.transforms.Normalize` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation inplace.
        Returns:
            Tensor: Normalized Tensor image.
        """

        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor


# Model Builder
class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'resnet18':
            orig_resnet = resnet18(pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet18(pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet50(pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet50(pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=NUM_CLASS,
                      weights='', use_softmax=False, drop_last_conv=False):
        arch = arch.lower()
        if arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                drop_last_conv=drop_last_conv)
        elif arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                drop_last_conv=drop_last_conv)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

    @staticmethod
    def get_decoder(weights_path, arch_encoder, arch_decoder, fc_dim, drop_last_conv, *arts, **kwargs):
        path = os.path.join(weights_path, 'ade20k', f'ade20k-{arch_encoder}-{arch_decoder}/decoder_epoch_20.pth')
        return ModelBuilder.build_decoder(arch=arch_decoder, fc_dim=fc_dim, weights=path, use_softmax=True, drop_last_conv=drop_last_conv)

    @staticmethod
    def get_encoder(weights_path, arch_encoder, arch_decoder, fc_dim, segmentation,
                    *arts, **kwargs):
        if segmentation:
          if not os.path.exists("encoder_epoch_20.pth"):
            import gdown
            url = 'https://drive.google.com/uc?id=1WdN7_wZbT5ZFz1nkca41ug8MX-sTF2rf'
            output = 'encoder_epoch_20.pth'
            gdown.download(url, output, quiet=False)
            print("encoder_epoch_20.pth downloaded")
          else:
            print("encoder_epoch_20.pth already downloaded")
          path = "encoder_epoch_20.pth"
        else:
            path = ''
        return ModelBuilder.build_encoder(arch=arch_encoder, fc_dim=fc_dim, weights=path)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class SegmentationModule(nn.Module):
    def __init__(self,
                 weights_path,
                 num_classes=150,
                 arch_encoder="resnet50dilated",
                 drop_last_conv=False,
                 net_enc=None,  # None for Default encoder
                 net_dec=None,  # None for Default decoder
                 encode=None,  # {None, 'binary', 'color', 'sky'}
                 use_default_normalization=False,
                 return_feature_maps=False,
                 return_feature_maps_level=3,  # {0, 1, 2, 3}
                 return_feature_maps_only=True,
                 **kwargs,
                 ):
        super().__init__()
        self.weights_path = weights_path
        self.drop_last_conv = drop_last_conv
        self.arch_encoder = arch_encoder
        if self.arch_encoder == "resnet50dilated":
            self.arch_decoder = "ppm_deepsup"
            self.fc_dim = 2048
        else:
            raise NotImplementedError(f"No such arch_encoder={self.arch_encoder}")
        model_builder_kwargs = dict(arch_encoder=self.arch_encoder,
                                    arch_decoder=self.arch_decoder,
                                    fc_dim=self.fc_dim,
                                    drop_last_conv=drop_last_conv,
                                    weights_path=self.weights_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = ModelBuilder.get_encoder(**model_builder_kwargs) if net_enc is None else net_enc
        self.decoder = ModelBuilder.get_decoder(**model_builder_kwargs) if net_dec is None else net_dec
        self.use_default_normalization = use_default_normalization
        self.default_normalization = NormalizeTensor(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])

        self.encode = encode

        self.return_feature_maps = return_feature_maps

        assert 0 <= return_feature_maps_level <= 3
        self.return_feature_maps_level = return_feature_maps_level

    def normalize_input(self, tensor):
        if tensor.min() < 0 or tensor.max() > 1:
            raise ValueError("Tensor should be 0..1 before using normalize_input")
        return self.default_normalization(tensor)

    @property
    def feature_maps_channels(self):
        return 256 * 2**(self.return_feature_maps_level)  # 256, 512, 1024, 2048

    def forward(self, img_data, segSize=None):
        if segSize is None:
            raise NotImplementedError("Please pass segSize param. By default: (300, 300)")

        fmaps = self.encoder(img_data, return_feature_maps=True)
        pred = self.decoder(fmaps, segSize=segSize)

        if self.return_feature_maps:
            return pred, fmaps
        # print("BINARY", img_data.shape, pred.shape)
        return pred

    def multi_mask_from_multiclass(self, pred, classes):
        def isin(ar1, ar2):
            return (ar1[..., None] == ar2).any(-1).float()
        return isin(pred, torch.LongTensor(classes).to(self.device))

    @staticmethod
    def multi_mask_from_multiclass_probs(scores, classes):
        res = None
        for c in classes:
            if res is None:
                res = scores[:, c]
            else:
                res += scores[:, c]
        return res

    def predict(self, tensor, imgSizes=(-1,),  # (300, 375, 450, 525, 600)
                segSize=None):
        """Entry-point for segmentation. Use this methods instead of forward
        Arguments:
            tensor {torch.Tensor} -- BCHW
        Keyword Arguments:
            imgSizes {tuple or list} -- imgSizes for segmentation input.
                default: (300, 450)
                original implementation: (300, 375, 450, 525, 600)

        """
        if segSize is None:
            segSize = tensor.shape[-2:]
        segSize = (tensor.shape[2], tensor.shape[3])
        with torch.no_grad():
            if self.use_default_normalization:
                tensor = self.normalize_input(tensor)
            scores = torch.zeros(1, NUM_CLASS, segSize[0], segSize[1]).to(self.device)
            features = torch.zeros(1, self.feature_maps_channels, segSize[0], segSize[1]).to(self.device)

            result = []
            for img_size in imgSizes:
                if img_size != -1:
                    img_data = F.interpolate(tensor.clone(), size=img_size)
                else:
                    img_data = tensor.clone()

                if self.return_feature_maps:
                    pred_current, fmaps = self.forward(img_data, segSize=segSize)
                else:
                    pred_current = self.forward(img_data, segSize=segSize)


                result.append(pred_current)
                scores = scores + pred_current / len(imgSizes)

                # Disclaimer: We use and aggregate only last fmaps: fmaps[3]
                if self.return_feature_maps:
                    features = features + F.interpolate(fmaps[self.return_feature_maps_level], size=segSize) / len(imgSizes)

            _, pred = torch.max(scores, dim=1)

            if self.return_feature_maps:
                return features

            return pred, result

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])

        if True:
            return edge.half()
        return edge.float()


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=NUM_CLASS, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 drop_last_conv=False):
        super().__init__()
        self.use_softmax = use_softmax
        self.drop_last_conv = drop_last_conv

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        if self.drop_last_conv:
            return ppm_out
        else:
            x = self.conv_last(ppm_out)

            if self.use_softmax:  # is True during inference
                x = nn.functional.interpolate(
                    x, size=segSize, mode='bilinear', align_corners=False)
                x = nn.functional.softmax(x, dim=1)
                return x

            # deep sup
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.dropout_deepsup(_)
            _ = self.conv_last_deepsup(_)

            x = nn.functional.log_softmax(x, dim=1)
            _ = nn.functional.log_softmax(_, dim=1)

            return (x, _)


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

# Resnet Dilated
class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super().__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]

# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False, drop_last_conv=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax
        self.drop_last_conv = drop_last_conv

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)

        if self.drop_last_conv:
            return x
        else:
            x = self.conv_last(x)

            if self.use_softmax:  # is True during inference
                x = nn.functional.interpolate(
                    x, size=segSize, mode='bilinear', align_corners=False)
                x = nn.functional.softmax(x, dim=1)
                return x

            # deep sup
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.conv_last_deepsup(_)

            x = nn.functional.log_softmax(x, dim=1)
            _ = nn.functional.log_softmax(_, dim=1)

            return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x



class PerceptualLoss(nn.Module):
    def __init__(self, normalize_inputs=True):
        super(PerceptualLoss, self).__init__()

        self.normalize_inputs = normalize_inputs
        self.mean_ = IMAGENET_MEAN
        self.std_ = IMAGENET_STD

        vgg = torchvision.models.vgg19(pretrained=True).features
        vgg_avg_pooling = []

        for weights in vgg.parameters():
            weights.requires_grad = False

        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)

        self.vgg = nn.Sequential(*vgg_avg_pooling)

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def partial_losses(self, input, target, mask=None):
        check_and_warn_input_range(target, 0, 1, 'PerceptualLoss target in partial_losses')

        # we expect input and target to be in [0, 1] range
        losses = []

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
            features_target = self.do_normalize_inputs(target)
        else:
            features_input = input
            features_target = target

        for layer in self.vgg[:30]:

            features_input = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':
                loss = F.mse_loss(features_input, features_target, reduction='none')

                if mask is not None:
                    cur_mask = F.interpolate(mask, size=features_input.shape[-2:],
                                             mode='bilinear', align_corners=False)
                    loss = loss * (1 - cur_mask)

                loss = loss.mean(dim=tuple(range(1, len(loss.shape))))
                losses.append(loss)

        return losses

    def forward(self, input, target, mask=None):
        losses = self.partial_losses(input, target, mask=mask)
        return torch.stack(losses).sum(dim=0)

    def get_global_features(self, input):
        check_and_warn_input_range(input, 0, 1, 'PerceptualLoss input in get_global_features')

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
        else:
            features_input = input

        features_input = self.vgg(features_input)
        return features_input


class ResNetPL(nn.Module):
    def __init__(self, weight=1,
                 weights_path=None, arch_encoder='resnet50dilated', segmentation=True):
        super().__init__()
        self.impl = ModelBuilder.get_encoder(weights_path=weights_path,
                                             arch_encoder=arch_encoder,
                                             arch_decoder='ppm_deepsup',
                                             fc_dim=2048,
                                             segmentation=segmentation)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight

    def forward(self, pred, target):
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)

        result = torch.stack([F.mse_loss(cur_pred, cur_target)
                              for cur_pred, cur_target
                              in zip(pred_feats, target_feats)]).sum() * self.weight
        return result