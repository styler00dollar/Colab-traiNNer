"""
torch_imports.py (9-3-20)
https://github.com/alfagao/DeOldify/blob/bc9d4562bf2014f5268f5c616ae31873577d9fde/fastai/torch_imports.py

conv_learner.py (9-3-20)
https://github.com/alfagao/DeOldify/blob/bc9d4562bf2014f5268f5c616ae31873577d9fde/fastai/conv_learner.py

model.py (9-3-20)
https://github.com/alfagao/DeOldify/blob/bc9d4562bf2014f5268f5c616ae31873577d9fde/fastai/model.py

modules.py (9-3-20)
https://github.com/alfagao/DeOldify/blob/bc9d4562bf2014f5268f5c616ae31873577d9fde/fasterai/modules.py

generators.py (9-3-20)
https://github.com/alfagao/DeOldify/blob/bc9d4562bf2014f5268f5c616ae31873577d9fde/fasterai/generators.py
"""

from abc import ABC, abstractmethod
from torchvision import transforms
from torch.nn.utils.spectral_norm import spectral_norm
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import pytorch_lightning as pl
from torchvision.models import vgg16_bn, vgg19_bn

def vgg16(pre): return children(vgg16_bn(pre))[0]
def vgg19(pre): return children(vgg19_bn(pre))[0]

def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]
"""
model_meta = {
    resnet18:[8,6], resnet34:[8,6], resnet50:[8,6], resnet101:[8,6], resnet152:[8,6],
    vgg16:[0,22], vgg19:[0,22],
    resnext50:[8,6], resnext101:[8,6], resnext101_64:[8,6],
    wrn:[8,6], inceptionresnet_2:[-2,9], inception_4:[-1,9],
    dn121:[0,7], dn161:[0,7], dn169:[0,7], dn201:[0,7],
}
"""

model_meta = {
    resnet18:[8,6], resnet34:[8,6], resnet50:[8,6], resnet101:[8,6], resnet152:[8,6],
    vgg16:[0,22], vgg19:[0,22],
}



class ConvBlock(pl.LightningModule):
    def __init__(self, ni:int, no:int, ks:int=3, stride:int=1, pad:int=None, actn:bool=True,
            bn:bool=True, bias:bool=True, sn:bool=False, leakyReLu:bool=False, self_attention:bool=False,
            inplace_relu:bool=True):
        super().__init__()
        if pad is None: pad = ks//2//stride

        if sn:
            layers = [spectral_norm(nn.Conv2d(ni, no, ks, stride, padding=pad, bias=bias))]
        else:
            layers = [nn.Conv2d(ni, no, ks, stride, padding=pad, bias=bias)]
        if actn:
            layers.append(nn.LeakyReLU(0.2, inplace=inplace_relu)) if leakyReLu else layers.append(nn.ReLU(inplace=inplace_relu))
        if bn:
            layers.append(nn.BatchNorm2d(no))
        if self_attention:
            layers.append(SelfAttention(no, 1))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class UpSampleBlock(pl.LightningModule):
    @staticmethod
    def _conv(ni:int, nf:int, ks:int=3, bn:bool=True, sn:bool=False, leakyReLu:bool=False):
        layers = [ConvBlock(ni, nf, ks=ks, sn=sn, bn=bn, actn=False, leakyReLu=leakyReLu)]
        return nn.Sequential(*layers)

    @staticmethod
    def _icnr(x:torch.Tensor, scale:int=2):
        init=nn.init.kaiming_normal_
        new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                                subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, scale ** 2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel

    def __init__(self, ni:int, nf:int, scale:int=2, ks:int=3, bn:bool=True, sn:bool=False, leakyReLu:bool=False):
        super().__init__()
        layers = []
        assert (math.log(scale,2)).is_integer()

        for i in range(int(math.log(scale,2))):
            layers += [UpSampleBlock._conv(ni, nf*4,ks=ks, bn=bn, sn=sn, leakyReLu=leakyReLu),
                nn.PixelShuffle(2)]
            if bn:
                layers += [nn.BatchNorm2d(nf)]

            ni = nf

        self.sequence = nn.Sequential(*layers)
        self._icnr_init()

    def _icnr_init(self):
        conv_shuffle = self.sequence[0][0].seq[0]
        kernel = UpSampleBlock._icnr(conv_shuffle.weight)
        conv_shuffle.weight.data.copy_(kernel)

    def forward(self, x):
        return self.sequence(x)


class UnetBlock(pl.LightningModule):
    def __init__(self, up_in:int , x_in:int , n_out:int, bn:bool=True, sn:bool=False, leakyReLu:bool=False,
            self_attention:bool=False, inplace_relu:bool=True):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = ConvBlock(x_in,  x_out,  ks=1, bn=False, actn=False, sn=sn, inplace_relu=inplace_relu)
        self.tr_conv = UpSampleBlock(up_in, up_out, 2, bn=bn, sn=sn, leakyReLu=leakyReLu)
        self.relu = nn.LeakyReLU(0.2, inplace=inplace_relu) if leakyReLu else nn.ReLU(inplace=inplace_relu)
        out_layers = []
        if bn:
            out_layers.append(nn.BatchNorm2d(n_out))
        if self_attention:
            out_layers.append(SelfAttention(n_out))
        self.out = nn.Sequential(*out_layers)


    def forward(self, up_p:int, x_p:int):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        x = torch.cat([up_p,x_p], dim=1)
        x = self.relu(x)
        return self.out(x)

class SaveFeatures():
    features=None
    def __init__(self, m:pl.LightningModule):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def remove(self):
        self.hook.remove()

class SelfAttention(pl.LightningModule):
    def __init__(self, in_channel:int, gain:int=1):
        super().__init__()
        self.query = self._spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),gain=gain)
        self.key = self._spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),gain=gain)
        self.value = self._spectral_init(nn.Conv1d(in_channel, in_channel, 1), gain=gain)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _spectral_init(self, module:pl.LightningModule, gain:int=1):
        nn.init.kaiming_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.zero_()

        return spectral_norm(module)

    def forward(self, input:torch.Tensor):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input
        return out


class GeneratorModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def set_trainable(self, trainable:bool):
        set_trainable(self, trainable)

    @abstractmethod
    def get_layer_groups(self, precompute:bool=False)->[]:
        pass

    @abstractmethod
    def forward(self, x_in:torch.Tensor, max_render_sz:int=400):
        pass

    def freeze_to(self, n:int):
        c=self.get_layer_groups()
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)

    def get_device(self):
        return next(self.parameters()).device


class AbstractUnet(GeneratorModule):
    def __init__(self, nf_factor:int=1, scale:int=1):
        super().__init__()
        assert (math.log(scale,2)).is_integer()
        self.rn, self.lr_cut = self._get_pretrained_resnet_base()
        ups = self._get_decoding_layers(nf_factor=nf_factor, scale=scale)
        self.relu = nn.ReLU()
        self.up1 = ups[0]
        self.up2 = ups[1]
        self.up3 = ups[2]
        self.up4 = ups[3]
        self.up5 = ups[4]
        self.out= nn.Sequential(ConvBlock(32*nf_factor, 3, ks=3, actn=False, bn=False, sn=True), nn.Tanh())

    @abstractmethod
    def _get_pretrained_resnet_base(self, layers_cut:int=0):
        pass

    @abstractmethod
    def _get_decoding_layers(self, nf_factor:int, scale:int):
        pass

    #Gets around irritating inconsistent halving coming from resnet
    def _pad(self, x:torch.Tensor, target:torch.Tensor, total_padh:int, total_padw:int)-> torch.Tensor:
        h = x.shape[2]
        w = x.shape[3]

        target_h = target.shape[2]*2
        target_w = target.shape[3]*2

        if h<target_h or w<target_w:
            padh = target_h-h if target_h > h else 0
            total_padh = total_padh + padh
            padw = target_w-w if target_w > w else 0
            total_padw = total_padw + padw
            return (F.pad(x, (0,padw,0,padh), "reflect",0), total_padh, total_padw)

        return (x, total_padh, total_padw)

    def _remove_padding(self, x:torch.Tensor, padh:int, padw:int)->torch.Tensor:
        if padw == 0 and padh == 0:
            return x

        target_h = x.shape[2]-padh
        target_w = x.shape[3]-padw
        return x[:,:,:target_h, :target_w]

    def _encode(self, x:torch.Tensor):
        x = self.rn[0](x)
        x = self.rn[1](x)
        x = self.rn[2](x)
        enc0 = x
        x = self.rn[3](x)
        x = self.rn[4](x)
        enc1 = x
        x = self.rn[5](x)
        enc2 = x
        x = self.rn[6](x)
        enc3 = x
        x = self.rn[7](x)
        return (x, enc0, enc1, enc2, enc3)

    def _decode(self, x:torch.Tensor, enc0:torch.Tensor, enc1:torch.Tensor, enc2:torch.Tensor, enc3:torch.Tensor):
        padh = 0
        padw = 0
        x = self.relu(x)
        enc3, padh, padw = self._pad(enc3, x, padh, padw)
        x = self.up1(x, enc3)
        enc2, padh, padw  = self._pad(enc2, x, padh, padw)
        x = self.up2(x, enc2)
        enc1, padh, padw  = self._pad(enc1, x, padh, padw)
        x = self.up3(x, enc1)
        enc0, padh, padw  = self._pad(enc0, x, padh, padw)
        x = self.up4(x, enc0)
        #This is a bit too much padding being removed, but I
        #haven't yet figured out a good way to determine what
        #exactly should be removed.  This is consistently more
        #than enough though.
        x = self.up5(x)
        x = self.out(x)
        x = self._remove_padding(x, padh, padw)
        return x

    def forward(self, x:torch.Tensor):
        x, enc0, enc1, enc2, enc3 = self._encode(x)
        x = self._decode(x, enc0, enc1, enc2, enc3)
        return x

    def get_layer_groups(self, precompute:bool=False)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]

    def close(self):
        for sf in self.sfs:
            sf.remove()


class Unet34(AbstractUnet):
    def __init__(self, nf_factor:int=1, scale:int=1):
        super().__init__(nf_factor=nf_factor, scale=scale)

    def _get_pretrained_resnet_base(self, layers_cut:int=0):
        f = resnet34
        cut,lr_cut = model_meta[f]
        cut-=layers_cut
        layers = cut_model(f(True), cut)
        return nn.Sequential(*layers), lr_cut

    def _get_decoding_layers(self, nf_factor:int, scale:int):
        self_attention=True
        bn=True
        sn=True
        leakyReLu=False
        layers = []
        layers.append(UnetBlock(512,256,512*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,128,512*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,64,512*nf_factor, sn=sn, self_attention=self_attention, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,64,256*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UpSampleBlock(256*nf_factor, 32*nf_factor, 2*scale, sn=sn, leakyReLu=leakyReLu, bn=bn))
        return layers


class Unet101(AbstractUnet):
    def __init__(self, nf_factor:int=1, scale:int=1):
        super().__init__(nf_factor=nf_factor, scale=scale)

    def _get_pretrained_resnet_base(self, layers_cut:int=0):
        f = resnet101
        cut,lr_cut = model_meta[f]
        cut-=layers_cut
        layers = cut_model(f(True), cut)
        return nn.Sequential(*layers), lr_cut

    def _get_decoding_layers(self, nf_factor:int, scale:int):
        self_attention=True
        bn=True
        sn=True
        leakyReLu=False
        layers = []
        layers.append(UnetBlock(2048,1024,512*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,512,512*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,256,512*nf_factor, sn=sn, self_attention=self_attention, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,64,256*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UpSampleBlock(256*nf_factor, 32*nf_factor, 2*scale, sn=sn, leakyReLu=leakyReLu, bn=bn))
        return layers

class Unet152(AbstractUnet):
    def __init__(self, nf_factor:int=1, scale:int=1):
        super().__init__(nf_factor=nf_factor, scale=scale)

    def _get_pretrained_resnet_base(self, layers_cut:int=0):
        f = resnet152
        cut,lr_cut = model_meta[f]
        cut-=layers_cut
        layers = cut_model(f(True), cut)
        return nn.Sequential(*layers), lr_cut

    def _get_decoding_layers(self, nf_factor:int, scale:int):
        self_attention=True
        bn=True
        sn=True
        leakyReLu=False
        layers = []
        layers.append(UnetBlock(2048,1024,512*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,512,512*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,256,512*nf_factor, sn=sn, self_attention=self_attention, leakyReLu=leakyReLu, bn=bn))
        layers.append(UnetBlock(512*nf_factor,64,256*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn))
        layers.append(UpSampleBlock(256*nf_factor, 32*nf_factor, 2*scale, sn=sn, leakyReLu=leakyReLu, bn=bn))
        return layers
