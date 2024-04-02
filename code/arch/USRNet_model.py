from __future__ import absolute_import

import logging
from collections import OrderedDict

import torch
import torch.nn as nn

import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger("base")

from . import losses
from . import optimizers
from . import schedulers
from . import swa

from dataops.batchaug import BatchAug
from dataops.filters import FilterHigh, FilterLow  # , FilterX

load_amp = hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")
if load_amp:
    from torch.cuda.amp import autocast, GradScaler

    logger.info("AMP library available")
else:
    logger.info("AMP library not available")


class nullcast:
    # nullcontext:
    # https://github.com/python/cpython/commit/0784a2e5b174d2dbf7b144d480559e650c5cf64c
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *excinfo):
        pass


class USRNetmodel(BaseModel):
    def __init__(self, opt):
        super(USRNetmodel, self).__init__(opt)
        train_opt = opt["train"]

        # set if data should be normalized (-1,1) or not (0,1)
        if self.is_train:
            opt["datasets"]["train"].get("znorm", False)

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
            if train_opt["gan_weight"]:
                self.netD = networks.define_D(opt).to(self.device)  # D
                self.netD.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            """
            Setup network cap
            """
            # define if the generator will have a final capping mechanism in the output
            self.outm = train_opt.get("finalcap", None)

            """
            Setup batch augmentations
            """
            self.mixup = train_opt.get("mixup", None)
            if self.mixup:
                # TODO: cutblur and cutout need model to be modified so LR and HR have the same dimensions (1x)
                self.mixopts = train_opt.get(
                    "mixopts", ["blend", "rgb", "mixup", "cutmix", "cutmixup"]
                )  # , "cutout", "cutblur"]
                self.mixprob = train_opt.get(
                    "mixprob", [1.0, 1.0, 1.0, 1.0, 1.0]
                )  # , 1.0, 1.0]
                self.mixalpha = train_opt.get(
                    "mixalpha", [0.6, 1.0, 1.2, 0.7, 0.7]
                )  # , 0.001, 0.7]
                self.aux_mixprob = train_opt.get("aux_mixprob", 1.0)
                self.aux_mixalpha = train_opt.get("aux_mixalpha", 1.2)
                self.mix_p = train_opt.get("mix_p", None)

            """
            Setup frequency separation
            """
            self.fs = train_opt.get("fs", None)
            self.f_low = None
            self.f_high = None
            if self.fs:
                lpf_type = train_opt.get("lpf_type", "average")
                hpf_type = train_opt.get("hpf_type", "average")
                self.f_low = FilterLow(filter_type=lpf_type).to(self.device)
                self.f_high = FilterHigh(filter_type=hpf_type).to(self.device)

            """
            Initialize losses
            """
            # Initialize the losses with the opt parameters
            # Generator losses:
            self.generatorlosses = losses.GeneratorLoss(opt, self.device)
            # TODO: show the configured losses names in logger
            # print(self.generatorlosses.loss_list)

            # Discriminator loss:
            if train_opt["gan_type"] and train_opt["gan_weight"]:
                self.cri_gan = True
                diffaug = train_opt.get("diffaug", None)
                dapolicy = None
                if diffaug:  # TODO: this if should not be necessary
                    dapolicy = train_opt.get(
                        "dapolicy", "color,translation,cutout"
                    )  # original
                self.adversarial = losses.Adversarial(
                    train_opt=train_opt,
                    device=self.device,
                    diffaug=diffaug,
                    dapolicy=dapolicy,
                )
                # D_update_ratio and D_init_iters are for WGAN
                self.D_update_ratio = train_opt.get("D_update_ratio", 1)
                self.D_init_iters = train_opt.get("D_init_iters", 0)
            else:
                self.cri_gan = False

            """
            Prepare optimizers
            """
            self.optGstep = False
            self.optDstep = False
            if self.cri_gan:
                (
                    self.optimizers,
                    self.optimizer_G,
                    self.optimizer_D,
                ) = optimizers.get_optimizers(
                    self.cri_gan,
                    self.netD,
                    self.netG,
                    train_opt,
                    logger,
                    self.optimizers,
                )
            else:
                self.optimizers, self.optimizer_G = optimizers.get_optimizers(
                    None, None, self.netG, train_opt, logger, self.optimizers
                )
                self.optDstep = True

            """
            Prepare schedulers
            """
            self.schedulers = schedulers.get_schedulers(
                optimizers=self.optimizers,
                schedulers=self.schedulers,
                train_opt=train_opt,
            )

            # Keep log in loss class instead?
            self.log_dict = OrderedDict()

            """
            Configure SWA
            """
            # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
            self.swa = opt.get("use_swa", False)
            if self.swa:
                self.swa_start_iter = train_opt.get("swa_start_iter", 0)
                # self.swa_start_epoch = train_opt.get('swa_start_epoch', None)
                swa_lr = train_opt.get("swa_lr", 0.0001)
                swa_anneal_epochs = train_opt.get("swa_anneal_epochs", 10)
                swa_anneal_strategy = train_opt.get("swa_anneal_strategy", "cos")
                # TODO: Note: This could be done in resume_training() instead, to prevent creating
                # the swa scheduler and model before they are needed
                self.swa_scheduler, self.swa_model = swa.get_swa(
                    self.optimizer_G,
                    self.netG,
                    swa_lr,
                    swa_anneal_epochs,
                    swa_anneal_strategy,
                )
                self.load_swa()  # load swa from resume state
                logger.info(
                    "SWA enabled. Starting on iter: {}, lr: {}".format(
                        self.swa_start_iter, swa_lr
                    )
                )

            """
            If using virtual batch
            """
            batch_size = opt["datasets"]["train"]["batch_size"]
            virtual_batch = opt["datasets"]["train"].get("virtual_batch_size", None)
            self.virtual_batch = (
                virtual_batch if virtual_batch >= batch_size else batch_size
            )
            self.accumulations = self.virtual_batch // batch_size
            self.optimizer_G.zero_grad()
            if self.cri_gan:
                self.optimizer_D.zero_grad()

            """
            Configure AMP
            """
            self.amp = load_amp and opt.get("use_amp", False)
            if self.amp:
                self.cast = autocast
                self.amp_scaler = GradScaler()
                logger.info("AMP enabled")
            else:
                self.cast = nullcast

            """
            Configure FreezeD
            """
            if self.cri_gan:
                loc = train_opt.get("freeze_loc", False)
                disc = opt["network_D"].get("which_model_D", False)
                if "discriminator_vgg" in disc and "fea" not in disc:
                    loc = (loc * 3) - 2
                elif "patchgan" in disc:
                    loc = (loc * 3) - 1
                # TODO: TMP, for now only tested with the vgg-like or patchgan discriminators
                if "discriminator_vgg" in disc or "patchgan" in disc:
                    self.feature_loc = loc
                    logger.info("FreezeD enabled")
                else:
                    self.feature_loc = None

        # print network
        """
        TODO:
        Network summary? Make optional with parameter
            could be an selector between traditional print_network() and summary()
        """
        # self.print_network() #TODO

    def feed_data(self, data, need_HR=True):
        # LR images
        self.var_L = data["LR"].to(self.device)
        self.ds_kernel = data["ds_kernel"].to(self.device)
        self.sigma = data["sigma"].to(self.device)

        if need_HR:  # train or val
            # HR images
            self.var_H = data["HR"].to(self.device)
            # discriminator references
            input_ref = data.get("ref", data["HR"])
            self.var_ref = input_ref.to(self.device)

    def feed_data_batch(self, data, need_HR=True):
        # LR
        self.var_L = data

    def optimize_parameters(self, step):
        # G
        # freeze discriminator while generator is trained to prevent BP
        if self.cri_gan:
            self.requires_grad(self.netD, flag=False, net_type="D")

        # batch (mixup) augmentations
        aug = None
        if self.mixup:
            self.var_H, self.var_L, mask, aug = BatchAug(
                self.var_H,
                self.var_L,
                self.mixopts,
                self.mixprob,
                self.mixalpha,
                self.aux_mixprob,
                self.aux_mixalpha,
                self.mix_p,
            )

        ### Network forward, generate SR
        with self.cast():
            # if self.outm: #if the model has the final activation option
            #    self.fake_H = self.netG(self.var_L, outm=self.outm)
            # else: #regular models without the final activation option
            self.fake_H = self.netG(self.var_L, self.ds_kernel, 4, self.sigma)
        # /with self.cast():

        # batch (mixup) augmentations
        # cutout-ed pixels are discarded when calculating loss by masking removed pixels
        if aug == "cutout":
            self.fake_H, self.var_H = self.fake_H * mask, self.var_H * mask

        l_g_total = 0
        """
        Calculate and log losses
        """
        loss_results = []
        # training generator and discriminator
        # update generator (on its own if only training generator or alternatively if training GAN)
        if (self.cri_gan is not True) or (
            step % self.D_update_ratio == 0 and step > self.D_init_iters
        ):
            with self.cast():  # Casts operations to mixed precision if enabled, else nullcontext
                # regular losses
                loss_results, self.log_dict = self.generatorlosses(
                    self.fake_H, self.var_H, self.log_dict, self.f_low
                )
                l_g_total += sum(loss_results) / self.accumulations

                if self.cri_gan:
                    # adversarial loss
                    l_g_gan = self.adversarial(
                        self.fake_H,
                        self.var_ref,
                        netD=self.netD,
                        stage="generator",
                        fsfilter=self.f_high,
                    )  # (sr, hr)
                    self.log_dict["l_g_gan"] = l_g_gan.item()
                    l_g_total += l_g_gan / self.accumulations

            # /with self.cast():

            if self.amp:
                # call backward() on scaled loss to create scaled gradients.
                self.amp_scaler.scale(l_g_total).backward()
            else:
                l_g_total.backward()

            # only step and clear gradient if virtual batch has completed
            if (step + 1) % self.accumulations == 0:
                if self.amp:
                    # unscale gradients of the optimizer's params, call
                    # optimizer.step() if no infs/NaNs in gradients, else, skipped
                    self.amp_scaler.step(self.optimizer_G)
                    # Update GradScaler scale for next iteration.
                    self.amp_scaler.update()
                    # TODO: remove. for debugging AMP
                    # print("AMP Scaler state dict: ", self.amp_scaler.state_dict())
                else:
                    self.optimizer_G.step()
                self.optimizer_G.zero_grad()
                self.optGstep = True

        if self.cri_gan:
            # update discriminator
            if isinstance(self.feature_loc, int):
                # unfreeze all D
                self.requires_grad(self.netD, flag=True)
                # then freeze up to the selected layers
                for loc in range(self.feature_loc):
                    self.requires_grad(self.netD, False, target_layer=loc, net_type="D")
            else:
                # unfreeze discriminator
                self.requires_grad(self.netD, flag=True)

            l_d_total = 0

            with self.cast():  # Casts operations to mixed precision if enabled, else nullcontext
                l_d_total, gan_logs = self.adversarial(
                    self.fake_H,
                    self.var_ref,
                    netD=self.netD,
                    stage="discriminator",
                    fsfilter=self.f_high,
                )  # (sr, hr)

                for g_log in gan_logs:
                    self.log_dict[g_log] = gan_logs[g_log]

                l_d_total /= self.accumulations
            # /with autocast():

            if self.amp:
                # call backward() on scaled loss to create scaled gradients.
                self.amp_scaler.scale(l_d_total).backward()
            else:
                l_d_total.backward()

            # only step and clear gradient if virtual batch has completed
            if (step + 1) % self.accumulations == 0:
                if self.amp:
                    # unscale gradients of the optimizer's params, call
                    # optimizer.step() if no infs/NaNs in gradients, else, skipped
                    self.amp_scaler.step(self.optimizer_D)
                    # Update GradScaler scale for next iteration.
                    self.amp_scaler.update()
                else:
                    self.optimizer_D.step()
                self.optimizer_D.zero_grad()
                self.optDstep = True

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            # if self.is_train:
            # self.fake_H = self.netG(self.var_L)
            self.fake_H = self.netG(self.var_L, self.ds_kernel, 4, self.sigma)
            # else:
            # self.fake_H = self.netG(self.var_L, isTest=True)
            # self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict["LR"] = self.var_L.detach()[0].float().cpu()
        out_dict["SR"] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict["HR"] = self.var_H.detach()[0].float().cpu()
        # TODO for PPON ?
        # if get stages 1 and 2
        # out_dict['SR_content'] = ...
        # out_dict['SR_structure'] = ...
        return out_dict

    def get_current_visuals_batch(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict["LR"] = self.var_L.detach().float().cpu()
        out_dict["SR"] = self.fake_H.detach().float().cpu()
        if need_HR:
            out_dict["HR"] = self.var_H.detach().float().cpu()
        # TODO for PPON ?
        # if get stages 1 and 2
        # out_dict['SR_content'] = ...
        # out_dict['SR_structure'] = ...
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = "{} - {}".format(
                self.netG.__class__.__name__, self.netG.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.netG.__class__.__name__)

        logger.info(
            "Network G structure: {}, with parameters: {:,d}".format(net_struc_str, n)
        )
        logger.info(s)
        if self.is_train:
            # Discriminator
            if self.cri_gan:
                s, n = self.get_network_description(self.netD)
                if isinstance(self.netD, nn.DataParallel):
                    net_struc_str = "{} - {}".format(
                        self.netD.__class__.__name__,
                        self.netD.module.__class__.__name__,
                    )
                else:
                    net_struc_str = "{}".format(self.netD.__class__.__name__)

                logger.info(
                    "Network D structure: {}, with parameters: {:,d}".format(
                        net_struc_str, n
                    )
                )
                logger.info(s)

            # TODO: feature network is not being trained, is it necessary to visualize? Maybe just name?
            # maybe show the generatorlosses instead?
            """
            if self.generatorlosses.cri_fea:  # F, Perceptual Network
                #s, n = self.get_network_description(self.netF)
                s, n = self.get_network_description(self.generatorlosses.netF) #TODO
                #s, n = self.get_network_description(self.generatorlosses.loss_list.netF) #TODO
                if isinstance(self.generatorlosses.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.generatorlosses.netF.__class__.__name__,
                                                    self.generatorlosses.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.generatorlosses.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)
            """

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading pretrained model for G [{:s}] ...".format(load_path_G))
            strict = self.opt["network_G"].get("strict", None)
            self.load_network(load_path_G, self.netG, strict, model_type="G")
        if self.opt["is_train"] and self.opt["train"]["gan_weight"]:
            load_path_D = self.opt["path"]["pretrain_model_D"]
            if self.opt["is_train"] and load_path_D is not None:
                logger.info(
                    "Loading pretrained model for D [{:s}] ...".format(load_path_D)
                )
                strict = self.opt["network_D"].get("strict", None)
                self.load_network(load_path_D, self.netD, model_type="D")

    def load_swa(self):
        if self.opt["is_train"] and self.opt["use_swa"]:
            load_path_swaG = self.opt["path"]["pretrain_model_swaG"]
            if self.opt["is_train"] and load_path_swaG is not None:
                logger.info(
                    "Loading pretrained model for SWA G [{:s}] ...".format(
                        load_path_swaG
                    )
                )
                self.load_network(load_path_swaG, self.swa_model)

    def save(self, iter_step, latest=None, loader=None):
        self.save_network(self.netG, "G", iter_step, latest)
        if self.cri_gan:
            self.save_network(self.netD, "D", iter_step, latest)
        if self.swa:
            # when training with networks that use BN
            # # Update bn statistics for the swa_model only at the end of training
            # if not isinstance(iter_step, int): #TODO: not sure if it should be done only at the end
            self.swa_model = self.swa_model.cpu()
            torch.optim.swa_utils.update_bn(loader, self.swa_model)
            self.swa_model = self.swa_model.cuda()
            # Check swa BN statistics
            # for module in self.swa_model.modules():
            #     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            #         print(module.running_mean)
            #         print(module.running_var)
            #         print(module.momentum)
            #         break
            self.save_network(self.swa_model, "swaG", iter_step, latest)
