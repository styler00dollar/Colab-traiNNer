import yaml
import cv2
from loss.metrics import *
from torchvision.utils import save_image
import pytorch_lightning as pl
from init import weights_init
import os
import numpy as np
from tensorboardX import SummaryWriter

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    
writer = SummaryWriter(logdir=cfg['path']['log_path'])


class CustomTrainClass(pl.LightningModule):
    def __init__(self):
        super().__init__()

        ##################################################################

        from generator import CreateGenerator
        self.netG = CreateGenerator(cfg)

        if (cfg['path']['checkpoint_path'] is None
                and cfg['network_G']['netG'] != 'GLEAN'
                and cfg['network_G']['netG'] != 'srflow'
                and cfg['network_G']['netG'] != 'GFPGAN'):
            if self.global_step == 0:
                weights_init(self.netG, 'kaiming')
                print("Generator weight init complete.")

        ##################################################################

        if cfg['network_D']['netD'] != None:
            from discriminator import CreateDiscriminator
            self.netD = CreateDiscriminator(cfg)
            
            # only doing init, if not 'TranformerDiscriminator', 'EfficientNet',
            # 'ResNeSt', 'resnet', 'ViT', 'DeepViT', 'mobilenetV3'
            if cfg['network_D']['netD'] in \
                    ('resnet3d', 'NFNet', 'context_encoder', 'VGG', 'VGG_fea', 'Discriminator_VGG_128_SN',
                        'VGGFeatureExtractor', 'NLayerDiscriminator', 'MultiscaleDiscriminator',
                        'Discriminator_ResNet_128', 'ResNet101FeatureExtractor', 'MINCNet',
                        'PixelDiscriminator', 'ResNeSt', 'RepVGG', 'squeezenet', 'SwinTransformer'):
                if self.global_step == 0:
                    weights_init(self.netD, 'kaiming')
                    print("Discriminator weight init complete.")

        ##################################################################

        # loss
        from loss_calc import AllLoss
        self.loss = AllLoss(cfg)

        # metrics
        self.psnr_metric = PSNR()
        self.ssim_metric = SSIM()
        self.ae_metric = AE()
        self.mse_metric = MSE()

        # logging
        if 'PSNR' in cfg['train']['metrics']:
            self.val_psnr = []
        if 'SSIM' in cfg['train']['metrics']:
            self.val_ssim = []
        if 'MSE' in cfg['train']['metrics']:
            self.val_mse = []
        if 'LPIPS' in cfg['train']['metrics']:
            self.val_lpips = []

        self.iter_check = 0

        if (cfg['train']['KID_weight'] > 0 
                or cfg['train']['IS_weight'] > 0 
                or cfg['train']['FID_weight'] > 0 
                or cfg['train']['PR_weight'] > 0):
            from loss.inceptionV3 import fid_inception_v3
            self.piq_model = fid_inception_v3()
            self.piq_model = self.piq_model.cuda().eval()
            if cfg['train']['force_piq_fp16'] is True:
                self.piq_model = self.piq_model.half()

        if cfg['datasets']['train']['mode'] == 'DS_realesrgan':
            from data.realesrgan import RealESRGANDatasetApply
            self.RealESRGANDatasetApply = RealESRGANDatasetApply(self.device)

    def forward(self, image, masks):
        return self.netG(image, masks)

    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        # iteration count is sometimes broken, adding a check and manual increment
        # only increment if generator gets trained (loop gets called a second time for discriminator)
        if self.trainer.global_step != 0:
            if optimizer_idx == 0 and self.iter_check == self.trainer.global_step:
                self.trainer.global_step += 1
            self.iter_check = self.trainer.global_step
            
        # different networks require different data and have different data loaders
        # due to overlap, a second check for dataloader mode is needed
        
        # if more than one output, fills dict with data, otherwise give empty dict to loss calc
        other = dict()

        # if inpainting
        if cfg['network_G']['netG'] in \
            (   # real inpainting generators
                'lama', 'MST', 'MANet', 'context_encoder', 'DFNet', 'AdaFill', 'MEDFE',
                'RFR', 'LBAM', 'DMFN', 'Partial', 'RN', 'DSNet', 'DSNetRRDB',
                'DSNetDeoldify', 'EdgeConnect', 'CSA', 'deepfillv1', 'deepfillv2',
                'Adaptive', 'Global', 'Pluralistic', 'crfill', 'DeepDFNet',
                'pennet', 'FRRN', 'PRVS', 'CRA', 'atrous', 'lightweight_gan', 'CTSDG',
                # sr genrators
                "restormer", "SRVGGNetCompact", "ESRT", "swinir", 'lightweight_gan', 'RRDB_net',
                'GLEAN', 'GPEN', 'comodgan', 'GFPGAN', 'swinir2') and \
                cfg['datasets']['train']['mode'] in ('DS_inpaint', 'DS_inpaint_TF'):
            lr_image = train_batch[0]
            hr_image = train_batch[2]
            other['mask'] = train_batch[1]

            if cfg['network_G']['netG'] == 'PRVS' or cfg['network_G']['netG'] == 'CTSDG':
                other['edge'] = train_batch[3]
            if cfg['network_G']['netG'] == 'EdgeConnect':
                other['grayscale'] = train_batch[4]

        # if super resolution
        elif cfg['network_G']['netG'] in \
                ("restormer", "SRVGGNetCompact", "ESRT", "swinir", 'lightweight_gan', 'RRDB_net',
                 'GLEAN', 'GPEN', 'comodgan', 'ASRGAN', 'PPON', 'sr_resnet', 'PAN', 'sisr',
                 'USRNet', 'srflow', 'DFDNet', 'GFPGAN', 'GPEN', 'comodgan', 'ESRT',
                 'SRVGGNetCompact', 'swinir2') and cfg['datasets']['train']['mode'] in ('DS_lrhr', 'DS_realesrgan'):
            if cfg['datasets']['train']['mode'] == 'DS_realesrgan':
                lr_image, hr_image, other['gt'] = self.RealESRGANDatasetApply.forward(train_batch[0], train_batch[1], \
                    train_batch[2], train_batch[3], self.device)
                # hotfix: at the end of one epoch it can happen that only 3d tensor gets returned
                if lr_image.dim() == 3:
                    lr_image = lr_image.unsqueeze(0)
                    hr_image = hr_image.unsqueeze(0)
            else:
                lr_image = train_batch[1]
                hr_image = train_batch[2]
            if cfg['network_G']['netG'] == 'DFDNet':
                other['landmarks'] = train_batch[3]

        # if interpolation
        elif cfg['network_G']['netG'] in \
                        ("CDFI", "sepconv_enhanced", 'CAIN', 'rife', 'RRIN', 'ABME', 'EDSC'):
            other['hr_image1'] = train_batch[0]
            other['hr_image3'] = train_batch[1]
            hr_image = train_batch[2]

        # train generator
        ############################
        if cfg['network_G']['netG'] == 'CTSDG':
            # input_image, input_edge, mask
            out, other['projected_image'], other['projected_edge'] = self.netG(lr_image, other['edge'], other['mask'])

        if cfg['network_G']['netG'] in \
                ('lama', 'MST', 'MANet', 'context_encoder', 'DFNet', 'AdaFill', 'MEDFE',
                 'RFR', 'LBAM', 'DMFN', 'Partial', 'RN', 'RN', 'DSNet', 'DSNetRRDB',
                 'DSNetDeoldify'):
            # generate fake (1 output)
            out = self.netG(lr_image, other['mask'])

        ############################
        if cfg['network_G']['netG'] in ('deepfillv1', 'deepfillv2', 'Adaptive'):
            # generate fake (2 outputs)
            out, other['other_img'] = self.netG(lr_image, other['mask'])

        ############################
        # exotic generators
        # CSA
        if cfg['network_G']['netG'] == 'CSA':
            other['coarse_result'], out, other['csa'], other['csa_d'] = self.netG(lr_image, other['mask'])

        # EdgeConnect
        if cfg['network_G']['netG'] == 'EdgeConnect':
            out, other['other_img'] = self.netG(lr_image, other['edge'], other['grayscale'], other['mask'])

        # PVRS
        if cfg['network_G']['netG'] == 'PVRS':
            out, _, other['edge_small'], other['edge_big'] = self.netG(lr_image, other['mask'], other['edge'])

        # FRRN
        if cfg['network_G']['netG'] == 'FRRN':
            out, other['mid_x'], other['mid_mask'] = self.netG(lr_image, other['mask'])

        # if inpaint, masking, taking original content from HR
        if cfg['network_G']['netG'] in ('CTSDG', 'lama', 'MST', 'MANet', 'context_encoder', 
                'DFNet', 'AdaFill', 'MEDFE', 'RFR', 'LBAM', 'DMFN', 'Partial', 'RN', 'RN', 
                'DSNet', 'DSNetRRDB', 'DSNetDeoldify', 'deepfillv1', 'deepfillv2', 'Adaptive', 
                'CSA', 'EdgeConnect', 'PVRS', 'FRRN'):
            out = lr_image*other['mask']+out*(1-other['mask'])

        # deoldify
        if cfg['network_G']['netG'] == 'deoldify':
            out = self.netG(lr_image)

        ############################
        # if frame interpolation
        if cfg['network_G']['netG'] in ("CDFI", "sepconv_enhanced", 'CAIN', 'RRIN', 'ABME', 'EDSC'):
            out = self.netG(other['hr_image1'], other['hr_image3'])

        if cfg['network_G']['netG'] == 'rife':
            out, other['flow'] = self.netG(other['hr_image1'], other['hr_image3'], training=True)

        # ESRT / swinir / lightweight_gan / RRDB_net / GLEAN / GPEN / comodgan
        if cfg['network_G']['netG'] in \
                ("restormer", "SRVGGNetCompact", "ESRT", "swinir", 'lightweight_gan', 'RRDB_net',
                 'GLEAN', 'GPEN', 'comodgan', 'swinir2'):
            if cfg['datasets']['train']['mode'] in ('DS_inpaint', 'DS_inpaint_TF'):
                out = self.netG(torch.cat([lr_image, other['mask']], 1))
                out = lr_image*other['mask']+out*(1-other['mask'])
            else:
                # normal dataloader
                out = self.netG(lr_image)
                # # unpad images if using CEM
                if cfg['network_G']['CEM'] is True:
                    out = self.CEM_net.HR_unpadder(out)
                    hr_image = self.CEM_net.HR_unpadder(hr_image)

        # GFPGAN
        if cfg['network_G']['netG'] == 'GFPGAN':
            if cfg['datasets']['train']['mode'] in ('DS_inpaint', 'DS_inpaint_TF'):
                out, _ = self.netG(torch.cat([lr_image, other['mask']], 1))
                out = lr_image*other['mask']+out*(1-other['mask'])
            else:
                out, _ = self.netG(lr_image)

        if cfg['network_G']['netG'] == 'srflow':
            # freeze rrdb in the beginning
            if self.trainer.global_step < cfg['network_G']['freeze_iter']:
                self.netG.set_rrdb_training(False)
            else:
                self.netG.set_rrdb_training(True)
            z, nll, y_logits = self.netG(gt=hr_image, lr=lr_image, reverse=False)
            out, other['logdet'] = self.netG(lr=lr_image, z=z, eps_std=0, reverse=True, reverse_with_grad=True)
            # out = torch.clamp(out, 0, 1)  # forcing out to be between 0 and 1

        # DFDNet
        if cfg['network_G']['netG'] == 'DFDNet':
            out = self.netG(lr_image, part_locations=other['landmarks'])
            # range [-1, 1] to [0, 1]
            out = out + 1
            out = out - out.min()
            out = out / (out.max() - out.min())

        total_loss = self.loss(out, hr_image, writer, self.trainer.global_step, optimizer_idx, self.netD, other)

        return total_loss

    def configure_optimizers(self):
        if cfg['network_G']['finetune'] is True:
            input_G = self.netG.parameters()
        else:
            input_G = filter(lambda p: p.requires_grad, self.netG.parameters())

        from optimizer import CreateOptimizer

        if cfg['network_D']['netD'] is not None:
            input_D = self.netD.parameters()
            opt_g, opt_d = CreateOptimizer(cfg, input_G, input_D)
            return [opt_g, opt_d], []
        else:
            opt_g, _ = CreateOptimizer(cfg, input_G)
            return [opt_g], []

    def validation_step(self, train_batch, train_idx):
        # different networks require different data and
        # have different data loaders
        
        # if inpainting
        if cfg['network_G']['netG'] in \
                ('lama', 'MST', 'MANet', 'context_encoder', 'DFNet', 'AdaFill', 'MEDFE',
                 'RFR', 'LBAM', 'DMFN', 'Partial', 'RN', 'RN', 'DSNet', 'DSNetRRDB',
                 'DSNetDeoldify'):
            lr_image = train_batch[0]
            mask = train_batch[1]
            path = train_batch[2]
            edge = train_batch[3]
            grayscale = train_batch[4]
        # if super resolution
        elif cfg['network_G']['netG'] in \
                ("restormer", "SRVGGNetCompact", "ESRT", "swinir", 'lightweight_gan', 'RRDB_net',
                 'GLEAN', 'GPEN', 'comodgan', 'swinir2'):
            lr_image = train_batch[0]
            hr_image = train_batch[1]
            path = train_batch[2]
            if cfg['network_G']['netG'] == 'DFDNet':
                landmarks = train_batch[3]
        # if interpolation
        elif cfg['network_G']['netG'] in \
                        ("CDFI", "sepconv_enhanced", 'CAIN', 'rife', 'RRIN', 'ABME', 'EDSC'):
            hr_image1, hr_image3 = train_batch[0]
            path = train_batch[2]

        #########################

        if cfg['network_G']['netG'] == 'CTSDG':
            out, _, _ = self.netG(lr_image, edge, mask)
            out = lr_image*mask+out*(1-mask)

        # if frame interpolation
        if cfg['network_G']['netG'] in \
                ("CDFI", "sepconv_enhanced", 'CAIN', 'RRIN', 'ABME', 'EDSC'):
            out = self.netG(hr_image1, hr_image3)

        if cfg['network_G']['netG'] == 'rife':
            out, _ = self.netG(hr_image1, hr_image3, training=True)

        #########################

        if cfg['network_G']['netG'] in \
                ('lama', 'MST', 'MANet', 'context_encoder', 'aotgan', 'DFNet', 'AdaFill',
                 'MEDFE', 'RFR', 'LBAM', 'DMFN', 'Partial', 'RN', 'RN', 'DSNet',
                 'DSNetRRDB', 'DSNetDeoldify'):
            out = self(lr_image, mask)

        if cfg['network_G']['netG'] in ('deepfillv1', 'deepfillv2', 'Adaptive'):
            out, _ = self(lr_image, mask)

        # CSA
        if cfg['network_G']['netG'] == 'CSA':
            _, out, _, _ = self(lr_image, mask)

        # EdgeConnect
        if cfg['network_G']['netG'] == 'EdgeConnect':
            out, _ = self.netG(lr_image, edge, grayscale, mask)

        # PVRS
        if cfg['network_G']['netG'] == 'PVRS':
            out, _, _, _ = self.netG(lr_image, mask, edge)

        # FRRN
        if cfg['network_G']['netG'] == 'FRRN':
            out, _, _ = self(lr_image, mask)

        # if inpaint, masking, taking original content from HR
        if cfg['network_G']['netG'] in ('CTSDG', 'lama', 'MST', 'MANet', 'context_encoder', 
                'DFNet', 'AdaFill', 'MEDFE', 'RFR', 'LBAM', 'DMFN', 'Partial', 'RN', 'RN', 
                'DSNet', 'DSNetRRDB', 'DSNetDeoldify', 'deepfillv1', 'deepfillv2', 'Adaptive', 
                'CSA', 'EdgeConnect', 'PVRS', 'FRRN'):
            out = lr_image*mask+out*(1-mask)

        # deoldify
        if cfg['network_G']['netG'] == 'deoldify':
            out = self.netG(lr_image)

        ############################
        # ESRGAN / GLEAN / GPEN / comodgan / lightweight_gan / ESRT / SRVGGNetCompact
        if cfg['network_G']['netG'] in \
                ("restormer", "SRVGGNetCompact", "ESRT", "swinir", 'lightweight_gan', 
                 'RRDB_net', 'GLEAN', 'GPEN', 'comodgan', 'swinir2'):
            if cfg['datasets']['train']['mode'] in ('DS_inpaint', 'DS_inpaint_TF'):
                # masked test with inpaint dataloader
                out = self.netG(torch.cat([lr_image, mask], 1))
                out = lr_image*mask+out*(1-mask)
            else:
                # normal dataloader
                out = self.netG(lr_image)

        # GFPGAN
        if cfg['network_G']['netG'] == 'GFPGAN':
            if cfg['datasets']['train']['mode'] in ('DS_inpaint', 'DS_inpaint_TF'):
                # masked test with inpaint dataloader
                out, _ = self.netG(torch.cat([lr_image, mask], 1))
                out = lr_image*mask+out*(1-mask)
            else:
                out, _ = self.netG(lr_image)

        if cfg['network_G']['netG'] == 'srflow':
            from arch.SRFlowNet_arch import get_z
            # freeze rrdb in the beginning
            if self.trainer.global_step < cfg['network_G']['freeze_iter']:
                self.netG.set_rrdb_training(False)
            else:
                self.netG.set_rrdb_training(True)

            z = get_z(self, heat=0, seed=None, batch_size=lr_image.shape[0], lr_shape=lr_image.shape)
            out, logdet = self.netG(lr=lr_image, z=z, eps_std=0, reverse=True, reverse_with_grad=True)

        # DFDNet
        if cfg['network_G']['netG'] == 'DFDNet':
            out = self.netG(lr_image, part_locations=landmarks)
            # range [-1, 1] to [0, 1]
            out = out + 1
            out = out - out.min()
            out = out / (out.max() - out.min())

        # Validation metrics work, but they need an origial source image.
        if 'PSNR' in cfg['train']['metrics']:
            self.val_psnr.append(self.psnr_metric(hr_image, out).item())
        if 'SSIM' in cfg['train']['metrics']:
            self.val_ssim.append(self.ssim_metric(hr_image, out).item())
        if 'MSE' in cfg['train']['metrics']:
            self.val_mse.append(self.mse_metric(hr_image, out).item())
        if 'LPIPS' in cfg['train']['metrics']:
            self.val_lpips.append(self.PerceptualLoss(out, hr_image).item())

        validation_output = cfg['path']['validation_output_path']

        # path can contain multiple files, depending on the batch_size
        for f in path:
            # data is processed as a batch, to save indididual files, a counter is used
            counter = 0
            if not os.path.exists(os.path.join(validation_output, os.path.splitext(os.path.basename(f))[0])):
                os.makedirs(os.path.join(validation_output, os.path.splitext(os.path.basename(f))[0]))

            filename_with_extention = os.path.basename(f)
            filename = os.path.splitext(filename_with_extention)[0]

            # currently only supports batch_size 1
            if cfg['network_G']['netG'] in ("sepconv_enhanced", 'CAIN', 'rife'):
                out = out.data.mul(255).mul(255 / 255).clamp(0, 255).round()
                out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()  # *255
                out = out.astype(np.uint8)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(validation_output, filename,
                                 str(self.trainer.global_step) + '.webp'), out)
            else:
                save_image(
                    out[counter], os.path.join(validation_output, filename,
                                               str(self.trainer.global_step) + '.webp'))

            counter += 1

    def validation_epoch_end(self, val_step_outputs):

        if 'PSNR' in cfg['train']['metrics']:
            val_psnr = np.mean(self.val_psnr)
            writer.add_scalar('metrics/PSNR', val_psnr, self.trainer.global_step)
            self.val_psnr = []
        if 'SSIM' in cfg['train']['metrics']:
            val_ssim = np.mean(self.val_ssim)
            writer.add_scalar('metrics/SSIM', val_ssim, self.trainer.global_step)
            self.val_ssim = []
        if 'MSE' in cfg['train']['metrics']:
            val_mse = np.mean(self.val_mse)
            writer.add_scalar('metrics/MSE', val_mse, self.trainer.global_step)
            self.val_mse = []
        if 'LPIPS' in cfg['train']['metrics']:
            val_lpips = np.mean(self.val_lpips)
            writer.add_scalar('metrics/LPIPS', val_lpips, self.trainer.global_step)
            self.val_lpips = []
