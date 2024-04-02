"""
20-Mai-2020
https://github.com/victorca25/BasicSR/blob/14aced7d1049a283761c145f3cf300a94c6ac4b9/codes/models/modules/architectures/SRFlow/RRDBNet_arch.py
"""

import functools
import torch.nn as nn
import torch.nn.functional as F
from . import block as B
from arch.rrdb_arch import RRDBM

# from options.options import opt_get
import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


class RRDBNet(nn.Module):
    """Modified RRDBNet"""

    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDBM, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = B.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        ## SRFlow changes
        # self.opt = opt
        self.scale = scale
        ## upsampling
        if self.scale >= 8:
            self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.scale >= 16:
            self.upconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.scale >= 32:
            self.upconv5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        ## /SRFlow changes

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, get_steps=False):
        fea = self.conv_first(x)

        # block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        block_idxs = cfg["network_G"]["flow"]["stackRRDB"]["blocks"]
        block_results = {}

        for idx, m in enumerate(self.RRDB_trunk.children()):
            fea = m(fea)
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea

        trunk = self.trunk_conv(fea)

        last_lr_fea = fea + trunk

        fea_up2 = self.upconv1(
            F.interpolate(last_lr_fea, scale_factor=2, mode="nearest")
        )
        fea = self.lrelu(fea_up2)

        fea_up4 = self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
        fea = self.lrelu(fea_up4)

        fea_up8 = None
        fea_up16 = None
        fea_up32 = None

        if self.scale >= 8:
            fea_up8 = self.upconv3(F.interpolate(fea, scale_factor=2, mode="nearest"))
            fea = self.lrelu(fea_up8)
        if self.scale >= 16:
            fea_up16 = self.upconv4(F.interpolate(fea, scale_factor=2, mode="nearest"))
            fea = self.lrelu(fea_up16)
        if self.scale >= 32:
            fea_up32 = self.upconv5(F.interpolate(fea, scale_factor=2, mode="nearest"))
            fea = self.lrelu(fea_up32)

        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        results = {
            "last_lr_fea": last_lr_fea,
            "fea_up1": last_lr_fea,
            "fea_up2": fea_up2,
            "fea_up4": fea_up4,
            "fea_up8": fea_up8,
            "fea_up16": fea_up16,
            "fea_up32": fea_up32,
            "out": out,
        }

        # fea_up0_en = opt_get(self.opt, ['network_G', 'flow', 'fea_up0']) or False
        fea_up0_en = cfg["network_G"]["flow"]["fea_up0"]
        if fea_up0_en:
            results["fea_up0"] = F.interpolate(
                last_lr_fea,
                scale_factor=1 / 2,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
        # fea_upn1_en = opt_get(self.opt, ['network_G', 'flow', 'fea_up-1']) or False
        # fea_upn1_en = cfg['network_G']['flow']['fea_up-1'] # empty
        fea_upn1_en = False

        if fea_upn1_en:
            results["fea_up-1"] = F.interpolate(
                last_lr_fea,
                scale_factor=1 / 4,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )

        if get_steps:
            for k, v in block_results.items():
                results[k] = v
            return results
        else:
            return out
