from torch import nn as nn
from torch.nn import functional as F

# https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py
class SRVGGNetCompact(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=2,
        act_type="prelu",
        conv_mode=3,
        rrdb=True,
        rrdb_blocks=2,
        convtype="Conv2D",
    ):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.add_act()

        # the body structure
        for _ in range(num_conv):
            if conv_mode == 3:
                self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            elif conv_mode == 2:
                self.body.append(
                    nn.Conv2d(num_feat, num_feat, kernel_size=2, padding=1)
                )
                self.body.append(
                    nn.Conv2d(num_feat, num_feat, kernel_size=2, padding=0)
                )
            else:
                print("Invalid conv mode!")
            self.add_act()

        if rrdb:
            from .rrdb_arch import RRDB

            for _ in range(rrdb_blocks):
                self.body.append(
                    RRDB(
                        num_feat,
                        nr=3,
                        kernel_size=3,
                        gc=32,
                        stride=1,
                        bias=1,
                        pad_type="zero",
                        norm_type=None,
                        act_type="leakyrelu",
                        mode="CNA",
                        convtype=convtype,
                        spectral_norm=False,
                        gaussian_noise=False,
                        plus=False,
                    )
                )
            self.add_act()

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def add_act(self):
        # activation
        if self.act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif self.act_type == "prelu":
            activation = nn.PReLU(num_parameters=self.num_feat)
        elif self.act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

    def forward(self, x):
        out = x

        feature_maps = []
        for i in range(0, len(self.body)):
            out = self.body[i](out)
            feature_maps.append(out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        out += base

        return out, feature_maps
