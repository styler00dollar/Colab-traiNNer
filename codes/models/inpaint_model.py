from __future__ import absolute_import
import numpy as np
import os
import logging
from collections import OrderedDict
import cv2
import torch
import torch.nn as nn

import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')

from . import losses
from . import optimizers
from . import schedulers
from . import swa

from dataops.batchaug import BatchAug
from dataops.filters import FilterHigh, FilterLow #, FilterX

load_amp = (hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"))
if load_amp:
    from torch.cuda.amp import autocast, GradScaler
    logger.info('AMP library available')
else:
    logger.info('AMP library not available')



from torchvision.utils import save_image


class nullcast():
    #nullcontext:
    #https://github.com/python/cpython/commit/0784a2e5b174d2dbf7b144d480559e650c5cf64c
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *excinfo):
        pass


class inpaintModel(BaseModel):
    def __init__(self, opt):
        super(inpaintModel, self).__init__(opt)


        self.counter = 0

        train_opt = opt['train']

        # set if data should be normalized (-1,1) or not (0,1)
        if self.is_train:
            z_norm = opt['datasets']['train'].get('znorm', False)

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
            if train_opt['gan_weight']:
                self.netD = networks.define_D(opt).to(self.device)  # D
                self.netD.train()
        self.load()  # load G and D if needed
        self.which_model_G = opt['network_G']['which_model_G']

        # define losses, optimizer and scheduler
        if self.is_train:
            """
            Setup network cap
            """
            # define if the generator will have a final capping mechanism in the output
            self.outm = train_opt.get('finalcap', None)

            """
            Setup batch augmentations
            """
            self.mixup = train_opt.get('mixup', None)
            if self.mixup:
                #TODO: cutblur and cutout need model to be modified so LR and HR have the same dimensions (1x)
                self.mixopts = train_opt.get('mixopts', ["blend", "rgb", "mixup", "cutmix", "cutmixup"]) #, "cutout", "cutblur"]
                self.mixprob = train_opt.get('mixprob', [1.0, 1.0, 1.0, 1.0, 1.0]) #, 1.0, 1.0]
                self.mixalpha = train_opt.get('mixalpha', [0.6, 1.0, 1.2, 0.7, 0.7]) #, 0.001, 0.7]
                self.aux_mixprob = train_opt.get('aux_mixprob', 1.0)
                self.aux_mixalpha = train_opt.get('aux_mixalpha', 1.2)
                self.mix_p = train_opt.get('mix_p', None)

            """
            Setup frequency separation
            """
            self.fs = train_opt.get('fs', None)
            self.f_low = None
            self.f_high = None
            if self.fs:
                lpf_type = train_opt.get('lpf_type', "average")
                hpf_type = train_opt.get('hpf_type', "average")
                self.f_low = FilterLow(filter_type=lpf_type).to(self.device)
                self.f_high = FilterHigh(filter_type=hpf_type).to(self.device)

            """
            Initialize losses
            """
            #Initialize the losses with the opt parameters
            # Generator losses:
            self.generatorlosses = losses.GeneratorLoss(opt, self.device)
            # TODO: show the configured losses names in logger
            # print(self.generatorlosses.loss_list)

            # Discriminator loss:
            if train_opt['gan_type'] and train_opt['gan_weight']:
                self.cri_gan = True
                diffaug = train_opt.get('diffaug', None)
                dapolicy = None
                if diffaug: #TODO: this if should not be necessary
                    dapolicy = train_opt.get('dapolicy', 'color,translation,cutout') #original
                self.adversarial = losses.Adversarial(train_opt=train_opt, device=self.device, diffaug = diffaug, dapolicy = dapolicy)
                # D_update_ratio and D_init_iters are for WGAN
                self.D_update_ratio = train_opt.get('D_update_ratio', 1)
                self.D_init_iters = train_opt.get('D_init_iters', 0)
            else:
                self.cri_gan = False

            """
            Prepare optimizers
            """
            self.optGstep = False
            self.optDstep = False
            if self.cri_gan:
                self.optimizers, self.optimizer_G, self.optimizer_D = optimizers.get_optimizers(
                    self.cri_gan, self.netD, self.netG, train_opt, logger, self.optimizers)
            else:
                self.optimizers, self.optimizer_G = optimizers.get_optimizers(
                    None, None, self.netG, train_opt, logger, self.optimizers)
                self.optDstep = True

            """
            Prepare schedulers
            """
            self.schedulers = schedulers.get_schedulers(
                optimizers=self.optimizers, schedulers=self.schedulers, train_opt=train_opt)

            #Keep log in loss class instead?
            self.log_dict = OrderedDict()

            """
            Configure SWA
            """
            #https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
            self.swa = opt.get('use_swa', False)
            if self.swa:
                self.swa_start_iter = train_opt.get('swa_start_iter', 0)
                # self.swa_start_epoch = train_opt.get('swa_start_epoch', None)
                swa_lr = train_opt.get('swa_lr', 0.0001)
                swa_anneal_epochs = train_opt.get('swa_anneal_epochs', 10)
                swa_anneal_strategy = train_opt.get('swa_anneal_strategy', 'cos')
                #TODO: Note: This could be done in resume_training() instead, to prevent creating
                # the swa scheduler and model before they are needed
                self.swa_scheduler, self.swa_model = swa.get_swa(
                        self.optimizer_G, self.netG, swa_lr, swa_anneal_epochs, swa_anneal_strategy)
                self.load_swa() #load swa from resume state
                logger.info('SWA enabled. Starting on iter: {}, lr: {}'.format(self.swa_start_iter, swa_lr))

            """
            If using virtual batch
            """
            batch_size = opt["datasets"]["train"]["batch_size"]
            virtual_batch = opt["datasets"]["train"].get('virtual_batch_size', None)
            self.virtual_batch = virtual_batch if virtual_batch \
                >= batch_size else batch_size
            self.accumulations = self.virtual_batch // batch_size
            self.optimizer_G.zero_grad()
            if self.cri_gan:
                self.optimizer_D.zero_grad()

            """
            Configure AMP
            """
            self.amp = load_amp and opt.get('use_amp', False)
            if self.amp:
                self.cast = autocast
                self.amp_scaler =  GradScaler()
                logger.info('AMP enabled')
            else:
                self.cast = nullcast

        # print network
        """
        TODO:
        Network summary? Make optional with parameter
            could be an selector between traditional print_network() and summary()
        """
        #self.print_network() #TODO

    #https://github.com/Yukariin/DFNet/blob/master/data.py
    def random_mask(self, height=256, width=256,
                  min_stroke=1, max_stroke=4,
                  min_vertex=1, max_vertex=12,
                  min_brush_width_divisor=16, max_brush_width_divisor=10):

      mask = np.ones((height, width))

      min_brush_width = height // min_brush_width_divisor
      max_brush_width = height // max_brush_width_divisor
      max_angle = 2*np.pi
      num_stroke = np.random.randint(min_stroke, max_stroke+1)
      average_length = np.sqrt(height*height + width*width) / 8



      for _ in range(num_stroke):
          num_vertex = np.random.randint(min_vertex, max_vertex+1)
          start_x = np.random.randint(width)
          start_y = np.random.randint(height)

          for _ in range(num_vertex):
              angle = np.random.uniform(max_angle)
              length = np.clip(np.random.normal(average_length, average_length//2), 0, 2*average_length)
              brush_width = np.random.randint(min_brush_width, max_brush_width+1)
              end_x = (start_x + length * np.sin(angle)).astype(np.int32)
              end_y = (start_y + length * np.cos(angle)).astype(np.int32)

              cv2.line(mask, (start_y, start_x), (end_y, end_x), 0., brush_width)

              start_x, start_y = end_x, end_y
      if np.random.random() < 0.5:
          mask = np.fliplr(mask)
      if np.random.random() < 0.5:
          mask = np.flipud(mask)
      return torch.from_numpy(mask.reshape((1,)+mask.shape).astype(np.float32)).unsqueeze(0)

    def masking_images(self):
        mask = self.random_mask(height=self.var_L.shape[2], width=self.var_L.shape[3]).cuda()
        for i in range(self.var_L.shape[0]-1):
          mask = torch.cat([mask, self.random_mask(height=self.var_L.shape[2], width=self.var_L.shape[3]).cuda()], dim=0)

        #self.var_L=self.var_L * mask
        return self.var_L * mask, mask

    def masking_images_with_invert(self):
        mask = self.random_mask(height=self.var_L.shape[2], width=self.var_L.shape[3]).cuda()
        for i in range(self.var_L.shape[0]-1):
          mask = torch.cat([mask, self.random_mask(height=self.var_L.shape[2], width=self.var_L.shape[3]).cuda()], dim=0)

        #self.var_L=self.var_L * mask
        return self.var_L * mask, self.var_L * (1-mask), mask

    def feed_data(self, data, need_HR=True):
        # LR images
        if self.which_model_G == 'EdgeConnect':
          self.var_L = data['LR'].to(self.device)
          self.canny_data = data['img_HR_canny'].to(self.device)
          self.grayscale_data = data['img_HR_gray'].to(self.device)
          #self.mask = data['green_mask'].to(self.device)
        else:
          self.var_L = data['LR'].to(self.device)

        if need_HR:  # train or val
            # HR images
            self.var_H = data['HR'].to(self.device)
            # discriminator references
            input_ref = data.get('ref', data['HR'])
            self.var_ref = input_ref.to(self.device)

    def feed_data_batch(self, data, need_HR=True):
        # LR
        self.var_L = data

    def optimize_parameters(self, step):
        # G
        # freeze discriminator while generator is trained to prevent BP
        if self.cri_gan:
            for p in self.netD.parameters():
                p.requires_grad = False

        # batch (mixup) augmentations
        aug = None
        if self.mixup:
            self.var_H, self.var_L, mask, aug = BatchAug(
                self.var_H, self.var_L,
                self.mixopts, self.mixprob, self.mixalpha,
                self.aux_mixprob, self.aux_mixalpha, self.mix_p
                )

        if self.which_model_G == 'Pluralistic':
          # pluralistic needs the inpainted area as an image and not only the cut-out
          self.var_L, img_inverted, mask = self.masking_images_with_invert()
        else:
          self.var_L, mask = self.masking_images()

        ### Network forward, generate SR
        with self.cast():
              # normal
              if self.which_model_G == 'RFR' or self.which_model_G == 'LBAM' or self.which_model_G == 'DMFN' or self.which_model_G == 'partial' or self.which_model_G == 'Adaptive' or self.which_model_G == 'DFNet' or self.which_model_G == 'RN':
                self.fake_H = self.netG(self.var_L, mask)
              # 2 rgb images
              if self.which_model_G == 'pennet' or self.which_model_G == 'deepfillv1' or self.which_model_G == 'deepfillv2' or self.which_model_G == 'Global' or self.which_model_G == 'crfill' or self.which_model_G == 'DeepDFNet':
                self.fake_H, self.other_img = self.netG(self.var_L, mask)

              # special
              if self.which_model_G == 'Pluralistic':
                self.fake_H, self.kl_rec, self.kl_g = self.netG(self.var_L, img_inverted, mask)

              if self.which_model_G == 'EdgeConnect':
                self.fake_H, self.other_img = self.netG(self.var_L, self.canny_data, self.grayscale_data, mask)

              if self.which_model_G == 'FRRN':
                self.fake_H, mid_x, mid_mask = self.netG(self.var_L, mask)

        #/with self.cast():
        #self.fake_H = self.netG(self.var_L, mask)

        #self.counter += 1
        #save_image(mask, str(self.counter)+'mask_train.png')
        #save_image(self.fake_H, str(self.counter)+'fake_H_train.png')

        # batch (mixup) augmentations
        # cutout-ed pixels are discarded when calculating loss by masking removed pixels
        if aug == "cutout":
            self.fake_H, self.var_H = self.fake_H*mask, self.var_H*mask

        l_g_total = 0
        """
        Calculate and log losses
        """
        loss_results = []
        # training generator and discriminator
        # update generator (on its own if only training generator or alternatively if training GAN)
        if (self.cri_gan is not True) or (step % self.D_update_ratio == 0 and step > self.D_init_iters):
            with self.cast(): # Casts operations to mixed precision if enabled, else nullcontext
                # regular losses
                loss_results, self.log_dict = self.generatorlosses(self.fake_H, self.var_H, self.log_dict, self.f_low)

                # additional losses, in case a model does output more than a normal image
                ###############################
                # deepfillv2 / global
                if self.which_model_G == 'deepfillv2' or self.which_model_G == 'Global' or self.which_model_G == 'crfill':
                  L1Loss = nn.L1Loss()
                  l1_stage1 = L1Loss(self.other_img, self.var_H)

                  self.log_dict.update(l1_stage1=l1_stage1)
                  loss_results.append(l1_stage1)

                # edge-connect
                if self.which_model_G == 'EdgeConnect':
                  L1Loss = nn.L1Loss()
                  l1_edge = L1Loss(self.other_img, self.var_H)

                  self.log_dict.update(l1_edge=l1_edge)
                  loss_results.append(l1_edge)
                ###############################
                # csa
                if self.which_model_G == 'CSA':
                  #coarse_result, refine_result, csa, csa_d = g_model(masked, mask)
                  L1Loss = nn.L1Loss()
                  recon_loss = L1Loss(coarse_result, img) + L1Loss(refine_result, img)

                  from models.modules.csa_loss import ConsistencyLoss
                  cons = ConsistencyLoss()
                  cons_loss = cons(csa, csa_d, img, mask)

                  self.log_dict.update(recon_loss=recon_loss)
                  loss_results.append(recon_loss)
                  self.log_dict.update(cons_loss=cons_loss)
                  loss_results.append(cons_loss)
                ###############################
                # pluralistic (encoder kl loss)
                if self.which_model_G == 'Pluralistic':
                  loss_kl_rec = self.kl_rec.mean()
                  loss_kl_g = self.kl_g.mean()

                  self.log_dict.update(loss_kl_rec=loss_kl_rec)
                  loss_results.append(loss_kl_rec)
                  self.log_dict.update(loss_kl_g=loss_kl_g)
                  loss_results.append(loss_kl_g)
                ###############################
                # deepfillv1
                if self.which_model_G == 'deepfillv1':
                  from models.modules.deepfillv1_loss import ReconLoss
                  ReconLoss_ = ReconLoss(1,1,1,1)
                  reconstruction_loss = ReconLoss_(self.var_H, self.other_img, self.fake_H, mask)

                  self.log_dict.update(reconstruction_loss=reconstruction_loss)
                  loss_results.append(reconstruction_loss)
                ###############################
                # pennet
                if self.which_model_G == 'pennet':
                  L1Loss = nn.L1Loss()
                  if self.other_img is not None:
                    pyramid_loss = 0
                    for _, f in enumerate(self.other_img):
                      pyramid_loss += L1Loss(f, torch.nn.functional.interpolate(self.var_H, size=f.size()[2:4], mode='bilinear', align_corners=True))

                  self.log_dict.update(pyramid_loss=pyramid_loss)
                  loss_results.append(pyramid_loss)
                ###############################
                # FRRN
                if self.which_model_G == 'FRRN':
                  L1Loss = nn.L1Loss()
                  # generator step loss
                  for idx in range(len(mid_x) - 1):
                      mid_l1_loss = L1Loss(mid_x[idx] * mid_mask[idx], self.var_H * mid_mask[idx])

                  self.log_dict.update(mid_l1_loss=mid_l1_loss)
                  loss_results.append(mid_l1_loss)
                ###############################

                #for key, value in self.log_dict.items():
                #    print(key, value)

                l_g_total += sum(loss_results)/self.accumulations

                if self.cri_gan:
                    # adversarial loss
                    l_g_gan = self.adversarial(
                        self.fake_H, self.var_ref, netD=self.netD,
                        stage='generator', fsfilter = self.f_high) # (sr, hr)
                    self.log_dict['l_g_gan'] = l_g_gan.item()
                    l_g_total += l_g_gan/self.accumulations

            #/with self.cast():

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
                    #TODO: remove. for debugging AMP
                    #print("AMP Scaler state dict: ", self.amp_scaler.state_dict())
                else:
                    self.optimizer_G.step()
                self.optimizer_G.zero_grad()
                self.optGstep = True

        if self.cri_gan:
            # update discriminator
            # unfreeze discriminator
            for p in self.netD.parameters():
                p.requires_grad = True
            l_d_total = 0

            with self.cast(): # Casts operations to mixed precision if enabled, else nullcontext
                l_d_total, gan_logs = self.adversarial(
                    self.fake_H, self.var_ref, netD=self.netD,
                    stage='discriminator', fsfilter = self.f_high) # (sr, hr)

                for g_log in gan_logs:
                    self.log_dict[g_log] = gan_logs[g_log]

                l_d_total /= self.accumulations
            #/with autocast():

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



    def test(self, data):
        """
        # generating random mask for validation
        self.var_L, mask = self.masking_images()
        if self.which_model_G == 'Pluralistic':
          # pluralistic needs the inpainted area as an image and not only the cut-out
          self.var_L, img_inverted, mask = self.masking_images_with_invert()
        else:
          self.var_L, mask = self.masking_images()
        """

        self.mask = data['green_mask'].float().to(self.device).unsqueeze(0)
        self.var_L = self.var_L * self.mask
        #print("self.mask")
        #print(self.mask)
        #self.var_L = self.var_L.float().cuda()

        self.netG.eval()
        with torch.no_grad():
            if self.is_train:
              # normal
              if self.which_model_G == 'RFR' or self.which_model_G == 'LBAM' or self.which_model_G == 'DMFN' or self.which_model_G == 'partial' or self.which_model_G == 'Adaptive' or self.which_model_G == 'DFNet' or self.which_model_G == 'RN':
                self.fake_H = self.netG(self.var_L, self.mask)
              # 2 rgb images
              if self.which_model_G == 'pennet' or self.which_model_G == 'deepfillv1' or self.which_model_G == 'deepfillv2' or self.which_model_G == 'Global' or self.which_model_G == 'crfill' or self.which_model_G == 'DeepDFNet':
                self.fake_H, _ = self.netG(self.var_L, self.mask)

              # special
              if self.which_model_G == 'Pluralistic':
                self.fake_H, _, _ = self.netG(self.var_L, img_inverted, self.mask)

              if self.which_model_G == 'EdgeConnect':
                self.fake_H, _ = self.netG(self.var_L, self.canny_data, self.grayscale_data, self.mask)

              if self.which_model_G == 'FRRN':
                self.fake_H, _, _ = self.netG(self.var_L, self.mask)

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        #TODO for PPON ?
        #if get stages 1 and 2
            #out_dict['SR_content'] = ...
            #out_dict['SR_structure'] = ...
        return out_dict

    def get_current_visuals_batch(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach().float().cpu()
        out_dict['SR'] = self.fake_H.detach().float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach().float().cpu()
        #TODO for PPON ?
        #if get stages 1 and 2
            #out_dict['SR_content'] = ...
            #out_dict['SR_structure'] = ...
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Discriminator
            if self.cri_gan:
                s, n = self.get_network_description(self.netD)
                if isinstance(self.netD, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                    self.netD.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netD.__class__.__name__)

                logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

            #TODO: feature network is not being trained, is it necessary to visualize? Maybe just name?
            # maybe show the generatorlosses instead?
            '''
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
            '''

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            strict = self.opt['path'].get('strict', None)
            self.load_network(load_path_G, self.netG, strict)
        if self.opt['is_train'] and self.opt['train']['gan_weight']:
            load_path_D = self.opt['path']['pretrain_model_D']
            if self.opt['is_train'] and load_path_D is not None:
                logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
                self.load_network(load_path_D, self.netD)

    def load_swa(self):
        if self.opt['is_train'] and self.opt['use_swa']:
            load_path_swaG = self.opt['path']['pretrain_model_swaG']
            if self.opt['is_train'] and load_path_swaG is not None:
                logger.info('Loading pretrained model for SWA G [{:s}] ...'.format(load_path_swaG))
                self.load_network(load_path_swaG, self.swa_model)

    def save(self, iter_step, latest=None, loader=None):
        self.save_network(self.netG, 'G', iter_step, latest)
        if self.cri_gan:
            self.save_network(self.netD, 'D', iter_step, latest)
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
            self.save_network(self.swa_model, 'swaG', iter_step, latest)
