"""
20-Mai-2020
https://github.com/victorca25/BasicSR/blob/14aced7d1049a283761c145f3cf300a94c6ac4b9/codes/models/modules/architectures/SRFlowNet_arch.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from models.modules.architectures.SRFlow.RRDBNet_arch import RRDBNet
from arch.SRFlow.RRDBNet_arch import RRDBNet

# from models.modules.architectures.SRFlow.FlowUpsamplerNet import FlowUpsamplerNet
from arch.SRFlow.FlowUpsamplerNet import FlowUpsamplerNet

# import models.modules.architectures.glow.thops as thops
import arch.glow.thops as thops

# import models.modules.architectures.glow.flow as flow
import arch.glow.flow as flow

# from options.options import opt_get
import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


# srflow
def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
    if seed:
        torch.manual_seed(seed)
    # if opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']):
    if cfg["network_G"]["flow"]["split"]["enable"]:
        # C = self.netG.module.flowUpsamplerNet.C
        C = self.netG.flowUpsamplerNet.C
        # H = int(cfg['scale'] * lr_shape[2] // self.netG.module.flowUpsamplerNet.scaleH)
        H = int(cfg["scale"] * lr_shape[2] // self.netG.flowUpsamplerNet.scaleH)
        # W = int(cfg['scale'] * lr_shape[3] // self.netG.module.flowUpsamplerNet.scaleW)
        W = int(cfg["scale"] * lr_shape[3] // self.netG.flowUpsamplerNet.scaleW)
        size = (batch_size, C, H, W)
        z = torch.normal(mean=0, std=heat, size=size) if heat > 0 else torch.zeros(size)
    else:
        # L = opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
        L = cfg["network_G"]["flow"]["L"]
        fac = 2 ** (L - 3)
        z_size = int(self.lr_size // (2 ** (L - 3)))
        z = torch.normal(
            mean=0, std=heat, size=(batch_size, 3 * 8 * 8 * fac * fac, z_size, z_size)
        )
    return z


class SRFlowNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, K=None, step=None):
        super(SRFlowNet, self).__init__()

        # self.opt = opt
        # self.quant = 255 if opt_get(opt, ['datasets', 'train', 'quant']) is \
        #                    None else opt_get(opt, ['datasets', 'train', 'quant'])
        self.quant = 255
        self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale)
        # hidden_channels = opt_get(opt, ['network_G', 'flow', 'hidden_channels'])
        # hidden_channels = hidden_channels or 64
        hidden_channels = 64
        self.RRDB_training = True  # Default is true

        # train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        set_RRDB_to_train = False
        if set_RRDB_to_train:
            self.set_rrdb_training(True)

        # Note: using this will cause RRDB optimizer not to be created and has
        # to be added with add_optimizer_and_scheduler_RRDB
        # set_RRDB_to_train = opt_get(self.opt, ['network_G', 'train_RRDB'])  # False
        # set_RRDB_to_train = False if not set_RRDB_to_train else set_RRDB_to_train
        # self.set_rrdb_training(set_RRDB_to_train)

        self.flowUpsamplerNet = FlowUpsamplerNet(
            (160, 160, 3),
            hidden_channels,
            K,
            flow_coupling=cfg["network_G"]["flow"]["coupling"],
        )
        self.i = 0

    def set_rrdb_training(self, trainable):
        if self.RRDB_training != trainable:
            for p in self.RRDB.parameters():
                p.requires_grad = trainable
            self.RRDB_training = trainable
            return True
        return False

    def forward(
        self,
        gt=None,
        lr=None,
        z=None,
        eps_std=None,
        reverse=False,
        epses=None,
        reverse_with_grad=False,
        lr_enc=None,
        add_gt_noise=False,
        step=None,
        y_label=None,
    ):
        if not reverse:
            return self.normal_flow(
                gt,
                lr,
                epses=epses,
                lr_enc=lr_enc,
                add_gt_noise=add_gt_noise,
                step=step,
                y_onehot=y_label,
            )
        else:
            # assert lr.shape[0] == 1
            assert lr.shape[1] == 3
            # assert lr.shape[2] == 20
            # assert lr.shape[3] == 20
            # assert z.shape[0] == 1
            # assert z.shape[1] == 3 * 8 * 8
            # assert z.shape[2] == 20
            # assert z.shape[3] == 20
            if reverse_with_grad:
                return self.reverse_flow(
                    lr,
                    z,
                    y_onehot=y_label,
                    eps_std=eps_std,
                    epses=epses,
                    lr_enc=lr_enc,
                    add_gt_noise=add_gt_noise,
                )
            else:
                with torch.no_grad():
                    return self.reverse_flow(
                        lr,
                        z,
                        y_onehot=y_label,
                        eps_std=eps_std,
                        epses=epses,
                        lr_enc=lr_enc,
                        add_gt_noise=add_gt_noise,
                    )

    def normal_flow(
        self,
        gt,
        lr,
        y_onehot=None,
        epses=None,
        lr_enc=None,
        add_gt_noise=True,
        step=None,
    ):
        if lr_enc is None:
            lr_enc = self.rrdbPreprocessing(lr)

        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = thops.pixels(gt)

        z = gt

        if add_gt_noise:
            # Setup
            # noiseQuant = opt_get(self.opt, ['network_G', 'flow', 'augmentation', 'noiseQuant'], True)
            noiseQuant = cfg["network_G"]["flow"]["augmentation"]["noiseQuant"]
            if noiseQuant:
                z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / self.quant)
            logdet = logdet + float(-np.log(self.quant) * pixels)

        # Encode
        epses, logdet = self.flowUpsamplerNet(
            rrdbResults=lr_enc,
            gt=z,
            logdet=logdet,
            reverse=False,
            epses=epses,
            y_onehot=y_onehot,
        )

        objective = logdet.clone()

        if isinstance(epses, (list, tuple)):
            z = epses[-1]
        else:
            z = epses

        objective = objective + flow.GaussianDiag.logp(None, None, z)

        nll = (-objective) / float(np.log(2.0) * pixels)

        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet

    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr, get_steps=True)
        # block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        block_idxs = cfg["network_G"]["flow"]["stackRRDB"]["blocks"]
        if len(block_idxs) > 0:
            concat = torch.cat(
                [rrdbResults["block_{}".format(idx)] for idx in block_idxs], dim=1
            )

            # if opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'concat']) or False:
            if cfg["network_G"]["flow"]["stackRRDB"]["concat"] or False:
                keys = ["last_lr_fea", "fea_up1", "fea_up2", "fea_up4"]
                if "fea_up0" in rrdbResults.keys():
                    keys.append("fea_up0")
                if "fea_up-1" in rrdbResults.keys():
                    keys.append("fea_up-1")
                # if self.opt['scale'] >= 8:
                if cfg["scale"] >= 8:
                    keys.append("fea_up8")
                # if self.opt['scale'] == 16:
                if cfg["scale"] == 16:
                    keys.append("fea_up16")
                for k in keys:
                    h = rrdbResults[k].shape[2]
                    w = rrdbResults[k].shape[3]
                    rrdbResults[k] = torch.cat(
                        [rrdbResults[k], F.interpolate(concat, (h, w))], dim=1
                    )
        return rrdbResults

    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma**2)) * thops.sum(
            z**2, dim=[1, 2, 3]
        ) - z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real

    def reverse_flow(
        self, lr, z, y_onehot, eps_std, epses=None, lr_enc=None, add_gt_noise=True
    ):
        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        # pixels = thops.pixels(lr) * self.opt['scale'] ** 2
        pixels = thops.pixels(lr) * cfg["scale"] ** 2

        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)

        if lr_enc is None:
            lr_enc = self.rrdbPreprocessing(lr)

        x, logdet = self.flowUpsamplerNet(
            rrdbResults=lr_enc,
            z=z,
            eps_std=eps_std,
            reverse=True,
            epses=epses,
            logdet=logdet,
        )

        return x, logdet
