import yaml
import cv2
from loss.loss import FocalFrequencyLoss, feature_matching_loss, FrobeniusNormLoss, \
    LapLoss, CharbonnierLoss, GANLoss, GradientPenaltyLoss, HFENLoss, TVLoss, \
    GradientLoss, ElasticLoss, RelativeL1, L1CosineSim, ClipL1, MaskedL1Loss, \
    MultiscalePixelLoss, FFTloss, OFLoss, L1_regularization, ColorLoss, AverageLoss, \
    GPLoss, CPLoss, SPL_ComputeWithTrace, SPLoss, Contextual_Loss, StyleLoss
from loss.metrics import *
from torchvision.utils import save_image
from torch.autograd import Variable
import pytorch_lightning as pl
from piq import SSIMLoss, MultiScaleSSIMLoss, VIFLoss, FSIMLoss, GMSDLoss, \
    MultiScaleGMSDLoss, VSILoss, HaarPSILoss, MDSILoss, BRISQUELoss, PieAPP, \
    DISTS, IS, FID, KID, PR
from init import weights_init
import os
import numpy as np
from tensorboardX import SummaryWriter

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    
writer = SummaryWriter(logdir=cfg['path']['log_path'])

# diffaug import is global since self can't be used
if cfg['train']['augmentation_method'] == "diffaug":
    from loss.diffaug import DiffAugment


class CustomTrainClass(pl.LightningModule):
    def __init__(self):
        super().__init__()
        ############################
        # generators with one output, no AMP means nan loss during training
        if cfg['network_G']['netG'] == 'RRDB_net':
            from arch.rrdb_arch import RRDBNet
            self.netG = RRDBNet(
                in_nc=cfg['network_G']['in_nc'],
                out_nc=cfg['network_G']['out_nc'],
                nf=cfg['network_G']['nf'],
                nb=cfg['network_G']['nb'],
                gc=cfg['network_G']['gc'],
                upscale=cfg['scale'],
                norm_type=cfg['network_G']['norm_type'],
                act_type=cfg['network_G']['net_act'],
                mode=cfg['network_G']['mode'],
                upsample_mode=cfg['network_G']['upsample_mode'],
                convtype=cfg['network_G']['convtype'],
                finalact=cfg['network_G']['finalact'],
                gaussian_noise=cfg['network_G']['gaussian'],
                plus=cfg['network_G']['plus'],
                nr=cfg['network_G']['nr'])

        # DFNet
        elif cfg['network_G']['netG'] == 'DFNet':
            from arch.DFNet_arch import DFNet
            self.netG = DFNet(
                c_img=cfg['network_G']['c_img'],
                c_mask=cfg['network_G']['c_mask'],
                c_alpha=cfg['network_G']['c_alpha'],
                mode=cfg['network_G']['mode'],
                norm=cfg['network_G']['norm'],
                act_en=cfg['network_G']['act_en'],
                act_de=cfg['network_G']['act_de'],
                en_ksize=cfg['network_G']['en_ksize'],
                de_ksize=cfg['network_G']['de_ksize'],
                blend_layers=cfg['network_G']['blend_layers'],
                conv_type=cfg['network_G']['conv_type'])

        # AdaFill
        elif cfg['network_G']['netG'] == 'AdaFill':
            from arch.AdaFill_arch import InpaintNet
            self.netG = InpaintNet()

        # MEDFE (batch_size: 1, no AMP)
        elif cfg['network_G']['netG'] == 'MEDFE':
            from arch.MEDFE_arch import MEDFEGenerator
            self.netG = MEDFEGenerator()

        # RFR
        # conv_type = partial or deform
        # Warning: One testrun with deform resulted in Nan errors after ~60k iterations. It is also very slow.
        # 'partial' is recommended, since this is what the official implementation does use.
        elif cfg['network_G']['netG'] == 'RFR':
            from arch.RFR_arch import RFRNet
            self.netG = RFRNet(conv_type=cfg['network_G']['conv_type'])

        # LBAM
        elif cfg['network_G']['netG'] == 'LBAM':
            from arch.LBAM_arch import LBAMModel
            self.netG = LBAMModel(inputChannels=cfg['network_G']['inputChannels'],
                                  outputChannels=cfg['network_G']['outputChannels'])

        # DMFN
        elif cfg['network_G']['netG'] == 'DMFN':
            from arch.DMFN_arch import InpaintingGenerator
            self.netG = InpaintingGenerator(in_nc=4, out_nc=3, nf=64, n_res=8,
                                            norm='in', activation='relu')

        # partial
        elif cfg['network_G']['netG'] == 'Partial':
            from arch.partial_arch import Model
            self.netG = Model()

        # RN
        elif cfg['network_G']['netG'] == 'RN':
            from arch.RN_arch import G_Net, rn_initialize_weights
            self.netG = G_Net(
                input_channels=cfg['network_G']['input_channels'], 
                residual_blocks=cfg['network_G']['residual_blocks'], 
                threshold=cfg['network_G']['threshold'])
            # using rn init to avoid errors
            if self.global_step == 0:
                RN_arch = rn_initialize_weights(self.netG, scale=0.1)

        # DSNet
        elif cfg['network_G']['netG'] == 'DSNet':
            from arch.DSNet_arch import DSNet
            self.netG = DSNet(
                layer_size=cfg['network_G']['layer_sizenr'],
                input_channels=cfg['network_G']['input_channels'],
                upsampling_mode=cfg['network_G']['upsampling_mode'])

        # context_encoder
        elif cfg['network_G']['netG'] == 'context_encoder':
            from arch.context_encoder_arch import Net_G
            self.netG = Net_G()

        # MANet
        elif cfg['network_G']['netG'] == 'MANet':
            from arch.MANet_arch import PConvUNet
            self.netG = PConvUNet()

        # GPEN
        elif cfg['network_G']['netG'] == 'GPEN':
            from arch.GPEN_arch import FullGenerator
            self.netG = FullGenerator(
                input_channels=cfg['network_G']['input_channels'],
                style_dim=cfg['network_G']['style_dim'],
                n_mlp=cfg['network_G']['n_mlp'],
                channel_multiplier=cfg['network_G']['channel_multiplier'],
                blur_kernel=cfg['network_G']['blur_kernel'],
                lr_mlp=cfg['network_G']['lr_mlp'])

        # comodgan
        elif cfg['network_G']['netG'] == 'comodgan':
            from arch.comodgan_arch import Generator
            self.netG = Generator(
                dlatent_size=cfg['network_G']['dlatent_size'],
                num_channels=cfg['network_G']['num_channels'],
                resolution=cfg['network_G']['resolution'],
                fmap_base=cfg['network_G']['fmap_base'],
                fmap_decay=cfg['network_G']['fmap_decay'],
                fmap_min=cfg['network_G']['fmap_min'],
                fmap_max=cfg['network_G']['fmap_max'],
                randomize_noise=cfg['network_G']['randomize_noise'],
                architecture=cfg['network_G']['architecture'],
                nonlinearity=cfg['network_G']['nonlinearity'],
                resample_kernel=cfg['network_G']['resample_kernel'],
                fused_modconv=cfg['network_G']['fused_modconv'],
                pix2pix=cfg['network_G']['pix2pix'],
                dropout_rate=cfg['network_G']['dropout_rate'],
                cond_mod=cfg['network_G']['cond_mod'],
                style_mod=cfg['network_G']['style_mod'],
                noise_injection=cfg['network_G']['noise_injection'])

        elif cfg['network_G']['netG'] == 'swinir':
            from arch.swinir_arch import SwinIR
            self.netG = SwinIR(
                upscale=cfg['network_G']['upscale'],
                in_chans=cfg['network_G']['in_chans'],
                img_size=cfg['network_G']['img_size'],
                window_size=cfg['network_G']['window_size'],
                img_range=cfg['network_G']['img_range'],
                depths=cfg['network_G']['depths'],
                embed_dim=cfg['network_G']['embed_dim'],
                num_heads=cfg['network_G']['num_heads'],
                mlp_ratio=cfg['network_G']['mlp_ratio'],
                upsampler=cfg['network_G']['upsampler'],
                resi_connection=cfg['network_G']['resi_connection'])

        # Experimental

        # DSNetRRDB
        elif cfg['network_G']['netG'] == 'DSNetRRDB':
            from arch.experimental.DSNetRRDB_arch import DSNetRRDB
            self.netG = DSNetRRDB(layer_size=8, input_channels=3, upsampling_mode='nearest',
                                  in_nc=4, out_nc=3, nf=128, nb=8, gc=32, upscale=1,
                                  norm_type=None, act_type='leakyrelu', mode='CNA',
                                  upsample_mode='upconv', convtype='Conv2D', finalact=None,
                                  gaussian_noise=True, plus=False, nr=3)

        # DSNetDeoldify
        elif cfg['network_G']['netG'] == 'DSNetDeoldify':
            from arch.experimental.DSNetDeoldify_arch import DSNetDeoldify
            self.netG = DSNetDeoldify()

        elif cfg['network_G']['netG'] == 'lightweight_gan':
            from arch.experimental.lightweight_gan_arch import Generator
            self.netG = Generator(
                image_size=cfg['network_G']['image_size'],
                latent_dim=cfg['network_G']['latent_dim'],
                fmap_max=cfg['network_G']['fmap_max'],
                fmap_inverse_coef=cfg['network_G']['fmap_inverse_coef'],
                transparent=cfg['network_G']['transparent'],
                greyscale=cfg['network_G']['greyscale'],
                freq_chan_attn=cfg['network_G']['freq_chan_attn'])

        elif cfg['network_G']['netG'] == 'SimpleFontGenerator512':
            from arch.experimental.lightweight_gan_arch import SimpleFontGenerator512
            self.netG = SimpleFontGenerator512(
                image_size=cfg['network_G']['image_size'],
                latent_dim=cfg['network_G']['latent_dim'],
                fmap_max=cfg['network_G']['fmap_max'],
                fmap_inverse_coef=cfg['network_G']['fmap_inverse_coef'],
                transparent=cfg['network_G']['transparent'],
                greyscale=cfg['network_G']['greyscale'],
                freq_chan_attn=cfg['network_G']['freq_chan_attn'])

        elif cfg['network_G']['netG'] == 'SimpleFontGenerator256':
            from arch.experimental.lightweight_gan_arch import SimpleFontGenerator256
            self.netG = SimpleFontGenerator256(
                image_size=cfg['network_G']['image_size'],
                latent_dim=cfg['network_G']['latent_dim'],
                fmap_max=cfg['network_G']['fmap_max'],
                fmap_inverse_coef=cfg['network_G']['fmap_inverse_coef'],
                transparent=cfg['network_G']['transparent'],
                greyscale=cfg['network_G']['greyscale'],
                freq_chan_attn=cfg['network_G']['freq_chan_attn'])

        ############################

        # generators with two outputs

        # deepfillv1
        elif cfg['network_G']['netG'] == 'deepfillv1':
            from arch.deepfillv1_arch import InpaintSANet
            self.netG = InpaintSANet()

        # deepfillv2
        # conv_type = partial or deform
        elif cfg['network_G']['netG'] == 'deepfillv2':
            from arch.deepfillv2_arch import GatedGenerator
            self.netG = GatedGenerator(
                in_channels=cfg['network_G']['in_channels'],
                out_channels=cfg['network_G']['out_channels'],
                latent_channels=cfg['network_G']['latent_channels'],
                pad_type=cfg['network_G']['pad_type'],
                activation=cfg['network_G']['activation'],
                norm=cfg['network_G']['norm'],
                conv_type=cfg['network_G']['conv_type'])

        # Adaptive
        # [Warning] Adaptive does not like PatchGAN, Multiscale and ResNet.
        elif cfg['network_G']['netG'] == 'Adaptive':
            from arch.Adaptive_arch import PyramidNet
            self.netG = PyramidNet(
                in_channels=cfg['network_G']['in_channels'],
                residual_blocks=cfg['network_G']['residual_blocks'],
                init_weights=cfg['network_G']['init_weights'])

        ############################
        # exotic generators

        # Pluralistic
        elif cfg['network_G']['netG'] == 'Pluralistic':
            from arch.Pluralistic_arch import PluralisticGenerator
            self.netG = PluralisticGenerator(
                ngf_E=cfg['network_G']['ngf_E'],
                z_nc_E=cfg['network_G']['z_nc_E'],
                img_f_E=cfg['network_G']['img_f_E'],
                layers_E=cfg['network_G']['layers_E'],
                norm_E=cfg['network_G']['norm_E'],
                activation_E=cfg['network_G']['activation_E'],
                ngf_G=cfg['network_G']['ngf_G'],
                z_nc_G=cfg['network_G']['z_nc_G'],
                img_f_G=cfg['network_G']['img_f_G'],
                L_G=cfg['network_G']['L_G'],
                output_scale_G=cfg['network_G']['output_scale_G'],
                norm_G=cfg['network_G']['norm_G'],
                activation_G=cfg['network_G']['activation_G'])

        # EdgeConnect
        elif cfg['network_G']['netG'] == 'EdgeConnect':
            from arch.EdgeConnect_arch import EdgeConnectModel
            # conv_type_edge: 'normal' # normal | partial | deform (has no spectral_norm)
            self.netG = EdgeConnectModel(
                residual_blocks_edge=cfg['network_G']['residual_blocks_edge'],
                residual_blocks_inpaint=cfg['network_G']['residual_blocks_inpaint'],
                use_spectral_norm=cfg['network_G']['use_spectral_norm'],
                conv_type_edge=cfg['network_G']['conv_type_edge'],
                conv_type_inpaint=cfg['network_G']['conv_type_inpaint'])

        # FRRN
        elif cfg['network_G']['netG'] == 'FRRN':
            from arch.FRRN_arch import FRRNet
            self.netG = FRRNet()

        # PRVS
        elif cfg['network_G']['netG'] == 'PRVS':
            from arch.PRVS_arch import PRVSNet
            self.netG = PRVSNet()

        # CSA
        elif cfg['network_G']['netG'] == 'CSA':
            from arch.CSA_arch import InpaintNet
            self.netG = InpaintNet(
                c_img=cfg['network_G']['c_img'],
                norm=cfg['network_G']['norm'],
                act_en=cfg['network_G']['act_en'],
                act_de=cfg['network_G']['network_G'])

        # deoldify
        elif cfg['network_G']['netG'] == 'deoldify':
            from arch.Deoldify_arch import Unet34
            self.netG = Unet34()

        # GLEAN (does init itself)
        elif cfg['network_G']['netG'] == 'GLEAN':
            from arch.GLEAN_arch import GLEANStyleGANv2
            if cfg['network_G']['pretrained'] is False:
                self.netG = GLEANStyleGANv2(
                    in_size=cfg['network_G']['in_size'],
                    out_size=cfg['network_G']['out_size'],
                    img_channels=cfg['network_G']['img_channels'],
                    img_channels_out=cfg['network_G']['img_channels_out'],
                    rrdb_channels=cfg['network_G']['rrdb_channels'],
                    num_rrdbs=cfg['network_G']['num_rrdbs'],
                    style_channels=cfg['network_G']['style_channels'],
                    num_mlps=cfg['network_G']['num_mlps'],
                    channel_multiplier=cfg['network_G']['channel_multiplier'],
                    blur_kernel=cfg['network_G']['blur_kernel'],
                    lr_mlp=cfg['network_G']['lr_mlp'],
                    default_style_mode=cfg['network_G']['default_style_mode'],
                    eval_style_mode=cfg['network_G']['eval_style_mode'],
                    mix_prob=cfg['network_G']['mix_prob'],
                    pretrained=None,
                    bgr2rgb=cfg['network_G']['bgr2rgb'])
            else:
                # using stylegan pretrain
                self.netG = GLEANStyleGANv2(
                    in_size=cfg['network_G']['in_size'],
                    out_size=cfg['network_G']['out_size'],
                    img_channels=cfg['network_G']['img_channels'],
                    img_channels_out=cfg['network_G']['img_channels_out'],
                    rrdb_channels=cfg['network_G']['rrdb_channels'],
                    num_rrdbs=cfg['network_G']['num_rrdbs'],
                    style_channels=cfg['network_G']['style_channels'],
                    num_mlps=cfg['network_G']['num_mlps'],
                    channel_multiplier=cfg['network_G']['channel_multiplier'],
                    blur_kernel=cfg['network_G']['blur_kernel'],
                    lr_mlp=cfg['network_G']['lr_mlp'],
                    default_style_mode=cfg['network_G']['default_style_mode'],
                    eval_style_mode=cfg['network_G']['eval_style_mode'],
                    mix_prob=cfg['network_G']['mix_prob'],
                    pretrained=dict(
                        ckpt_path='http://download.openmmlab.com/mmgen/stylegan2/'
                                  'official_weights/stylegan2-ffhq-config-f-official_'
                                  '20210327_171224-bce9310c.pth', prefix='generator_ema'),
                    bgr2rgb=cfg['network_G']['bgr2rgb'])

        # srflow (weight init?)
        elif cfg['network_G']['netG'] == 'srflow':
            from arch.SRFlowNet_arch import SRFlowNet
            self.netG = SRFlowNet(
                in_nc=cfg['network_G']['in_nc'],
                out_nc=cfg['network_G']['out_nc'],
                nf=cfg['network_G']['nf'],
                nb=cfg['network_G']['nb'],
                scale=cfg['scale'],
                K=cfg['network_G']['flow']['K'],
                step=None)
            from arch.SRFlowNet_arch import get_z

        # DFDNet
        elif cfg['network_G']['netG'] == 'DFDNet':
            from arch.DFDNet_arch import UNetDictFace
            self.netG = UNetDictFace(64)

        # GFPGAN (error with init?)
        elif cfg['network_G']['netG'] == 'GFPGAN':
            from arch.GFPGAN_arch import GFPGANv1
            self.netG = GFPGANv1(
                input_channels=cfg['network_G']['input_channels'],
                output_channels=cfg['network_G']['output_channels'],
                out_size=cfg['network_G']['out_size'],
                num_style_feat=cfg['network_G']['num_style_feat'],
                channel_multiplier=cfg['network_G']['channel_multiplier'],
                resample_kernel=cfg['network_G']['resample_kernel'],
                decoder_load_path=cfg['network_G']['decoder_load_path'],
                fix_decoder=cfg['network_G']['fix_decoder'],
                num_mlp=cfg['network_G']['num_mlp'],
                lr_mlp=cfg['network_G']['lr_mlp'],
                input_is_latent=cfg['network_G']['input_is_latent'],
                different_w=cfg['network_G']['different_w'],
                narrow=cfg['network_G']['narrow'],
                sft_half=cfg['network_G']['sft_half'])

        elif cfg['network_G']['netG'] == 'CAIN':
            from arch.CAIN_arch import CAIN
            self.netG = CAIN(cfg['network_G']['depth'])

        elif cfg['network_G']['netG'] == 'rife':
            from arch.rife_arch import IFNet
            self.netG = IFNet()

        elif cfg['network_G']['netG'] == 'RRIN':
            from arch.RRIN_arch import Net
            self.netG = Net()

        elif cfg['network_G']['netG'] == 'ABME':
            from arch.ABME_arch import ABME
            self.netG = ABME()

        elif cfg['network_G']['netG'] == 'EDSC':
            from arch.EDSC_arch import Network
            self.netG = Network()

        elif cfg['network_G']['netG'] == 'CTSDG':
            from arch.CTSDG_arch import Generator
            self.netG = Generator()

        elif cfg['network_G']['netG'] == 'MST':
            from arch.MST_arch import InpaintGateGenerator
            self.netG = InpaintGateGenerator()

        elif cfg['network_G']['netG'] == 'lama':
            from arch.lama_arch import FFCResNetGenerator
            self.netG = FFCResNetGenerator(4, 3)

        elif cfg['network_G']['netG'] == "ESRT":
            from arch.ESRT_arch import ESRT
            self.netG = ESRT(
                hiddenDim=cfg['network_G']['hiddenDim'], 
                mlpDim=cfg['network_G']['mlpDim'], 
                scaleFactor=cfg['scale'])

        elif cfg['network_G']['netG'] == 'sepconv_enhanced':
            from arch.sepconv_enhanced_arch import Network
            self.netG = Network()

        elif cfg['network_G']['netG'] == "CDFI":
            from arch.CDFI_arch import AdaCoFNet
            self.netG = AdaCoFNet()

        elif cfg['network_G']['netG'] == "SRVGGNetCompact":
            from arch.SRVGGNetCompact_arch import SRVGGNetCompact
            self.netG = SRVGGNetCompact(
                num_in_ch=cfg['network_G']['num_in_ch'], 
                num_out_ch=cfg['network_G']['num_out_ch'], 
                num_feat=cfg['network_G']['num_feat'], 
                num_conv=cfg['network_G']['num_conv'], 
                upscale=cfg['network_G']['upscale'], 
                act_type=cfg['network_G']['act_type'])

        elif cfg['network_G']['netG'] == "restormer":
            from arch.restormer_arch import Restormer
            self.netG = Restormer(
                inp_channels=cfg['network_G']['inp_channels'],
                out_channels=cfg['network_G']['out_channels'],
                dim=cfg['network_G']['dim'],
                num_blocks=cfg['network_G']['num_blocks'],
                num_refinement_blocks=cfg['network_G']['num_refinement_blocks'],
                heads=cfg['network_G']['heads'],
                ffn_expansion_factor=cfg['network_G']['ffn_expansion_factor'],
                bias=cfg['network_G']['bias'],
                LayerNorm_type=cfg['network_G']['LayerNorm_type'])

        if (cfg['path']['checkpoint_path'] is None
                and cfg['network_G']['netG'] != 'GLEAN'
                and cfg['network_G']['netG'] != 'srflow'
                and cfg['network_G']['netG'] != 'GFPGAN'):
            if self.global_step == 0:
                weights_init(self.netG, 'kaiming')
                print("Generator weight init complete.")

        ############################

        if cfg['network_G']['CEM'] is True:
            CEM_conf = CEMnet.Get_CEM_Conf(cfg['scale'])
            CEM_conf.sigmoid_range_limit = cfg['network_G']['sigmoid_range_limit']
            if CEM_conf.sigmoid_range_limit:
                CEM_conf.input_range = [-1, 1] if z_norm else [0, 1]
            kernel = None  # note: could pass a kernel here, but None will use default cubic kernel
            self.CEM_net = CEMnet.CEMnet(CEM_conf, upscale_kernel=kernel)
            self.CEM_net.WrapArchitecture(only_padders=True)
            self.netG = self.CEM_net.WrapArchitecture(
                self.netG, training_patch_size=cfg['datasets']['train']['HR_size'])

        ############################

        # discriminators
        # size refers to input shape of tensor
        if cfg['network_D']['netD'] == 'context_encoder':
            from arch.discriminators import context_encoder
            self.netD = context_encoder()

        # VGG
        elif cfg['network_D']['netD'] == 'VGG':
            from arch.discriminators import Discriminator_VGG
            self.netD = Discriminator_VGG(
                size=cfg['network_D']['size'], 
                in_nc=cfg['network_D']['in_nc'], 
                base_nf=cfg['network_D']['base_nf'], 
                norm_type=cfg['network_D']['norm_type'], 
                act_type=cfg['network_D']['act_type'], 
                mode=cfg['network_D']['mode'], 
                convtype=cfg['network_D']['convtype'], 
                arch=cfg['network_D']['arch'])

        elif cfg['network_D']['netD'] == 'VGG_fea':
            from arch.discriminators import Discriminator_VGG_fea
            self.netD = Discriminator_VGG_fea(
                size=cfg['network_D']['size'], 
                in_nc=cfg['network_D']['in_nc'], 
                base_nf=cfg['network_D']['base_nf'], 
                norm_type=cfg['network_D']['norm_type'], 
                act_type=cfg['network_D']['act_type'], 
                mode=cfg['network_D']['mode'], 
                convtype=cfg['network_D']['convtype'],
                arch=cfg['network_D']['arch'], 
                spectral_norm=cfg['network_D']['spectral_norm'], 
                self_attention=cfg['network_D']['self_attention'],
                max_pool=cfg['network_D']['max_pool'], 
                poolsize=cfg['network_D']['poolsize'])

        elif cfg['network_D']['netD'] == 'Discriminator_VGG_128_SN':
            from arch.discriminators import Discriminator_VGG_128_SN
            self.netD = Discriminator_VGG_128_SN()

        elif cfg['network_D']['netD'] == 'VGGFeatureExtractor':
            from arch.discriminators import VGGFeatureExtractor
            self.netD = VGGFeatureExtractor(
                feature_layer=cfg['feature_layer']['feature_layer'],
                use_bn=cfg['network_D']['use_bn'],
                use_input_norm=cfg['network_D']['use_input_norm'],
                device=torch.device(cfg['network_D']['device']),
                z_norm=cfg['network_D']['z_norm'])

        # PatchGAN
        elif cfg['network_D']['netD'] == 'NLayerDiscriminator':
            from arch.discriminators import NLayerDiscriminator
            self.netD = NLayerDiscriminator(
                input_nc=cfg['network_D']['input_nc'],
                ndf=cfg['network_D']['ndf'],
                n_layers=cfg['network_D']['n_layers'],
                norm_layer=cfg['network_D']['norm_layer'],
                use_sigmoid=cfg['network_D']['use_sigmoid'],
                getIntermFeat=cfg['network_D']['getIntermFeat'],
                patch=cfg['network_D']['patch'],
                use_spectral_norm=cfg['network_D']['use_spectral_norm'])

        # Multiscale
        elif cfg['network_D']['netD'] == 'MultiscaleDiscriminator':
            from arch.discriminators import MultiscaleDiscriminator
            self.netD = MultiscaleDiscriminator(
                input_nc=cfg['network_D']['input_nc'],
                ndf=cfg['network_D']['ndf'],
                n_layers=cfg['network_D']['n_layers'],
                use_sigmoid=cfg['network_D']['use_sigmoid'],
                num_D=cfg['network_D']['num_D'],
                get_feats=cfg['network_D']['get_feats'])

        # ResNet
        # elif cfg['network_D']['netD'] == 'Discriminator_ResNet_128':
        #    from arch.discriminators import Discriminator_ResNet_128
        #    self.netD = Discriminator_ResNet_128(in_nc=cfg['network_D']['in_nc'],
        #    base_nf=cfg['network_D']['base_nf'],
        #    norm_type=cfg['network_D']['norm_type'],
        #    act_type=cfg['network_D']['act_type'],
        #    mode=cfg['network_D']['mode'])

        elif cfg['network_D']['netD'] == 'ResNet101FeatureExtractor':
            from arch.discriminators import ResNet101FeatureExtractor
            self.netD = ResNet101FeatureExtractor(
                use_input_norm=cfg['network_D']['use_input_norm'], 
                device=torch.device(cfg['network_D']['device']), 
                z_norm=cfg['network_D']['z_norm'])

        # MINC
        elif cfg['network_D']['netD'] == 'MINCNet':
            from arch.discriminators import MINCNet
            self.netD = MINCNet()

        # Pixel
        elif cfg['network_D']['netD'] == 'PixelDiscriminator':
            from arch.discriminators import PixelDiscriminator
            self.netD = PixelDiscriminator(
                input_nc=cfg['network_D']['input_nc'], 
                ndf=cfg['network_D']['ndf'], 
                norm_layer=cfg['network_D']['norm_layer'])

        # EfficientNet
        elif cfg['network_D']['netD'] == 'EfficientNet':
            from efficientnet_pytorch import EfficientNet
            self.netD = EfficientNet.from_pretrained(
                cfg['network_D']['EfficientNet_pretrain'], 
                num_classes=cfg['network_D']['num_classes'])

        # mobilenetV3
        elif cfg['network_D']['netD'] == "mobilenetV3":
            from arch.mobilenetv3_arch import MobileNetV3
            self.netD = MobileNetV3(
                n_class=cfg['network_D']['n_class'], 
                mode=cfg['network_D']['mode'], 
                input_size=cfg['network_D']['input_size'])

        # resnet
        elif cfg['network_D']['netD'] == 'resnet':
            if cfg['network_D']['pretrain'] is False:
                if cfg['network_D']['resnet_arch'] == 'resnet50':
                    from arch.resnet_arch import resnet50
                    self.netD = resnet50(num_classes=cfg['network_D']['num_classes'], 
                                         pretrain=cfg['network_D']['pretrain'])
                elif cfg['network_D']['resnet_arch'] == 'resnet101':
                    from arch.resnet_arch import resnet101
                    self.netD = resnet101(num_classes=cfg['network_D']['num_classes'], 
                                          pretrain=cfg['network_D']['pretrain'])
                elif cfg['network_D']['resnet_arch'] == 'resnet152':
                    from arch.resnet_arch import resnet152
                    self.netD = resnet152(num_classes=cfg['network_D']['num_classes'], 
                                          pretrain=cfg['network_D']['pretrain'])
                weights_init(self.netG, 'kaiming')
                print("Discriminator weight init complete.")

            if cfg['network_D']['pretrain'] is True:
                # loading a pretrained network does not work by default, the amount of classes
                # needs to be adjusted in the final layer
                import torchvision.models as models
                if cfg['network_D']['resnet_arch'] == 'resnet50':
                    pretrained_model = models.resnet50(pretrained=True)
                elif cfg['network_D']['resnet_arch'] == 'resnet101':
                    pretrained_model = models.resnet101(pretrained=True)
                elif cfg['network_D']['resnet_arch'] == 'resnet152':
                    pretrained_model = models.resnet152(pretrained=True)

                IN_FEATURES = pretrained_model.fc.in_features
                OUTPUT_DIM = cfg['network_D']['num_classes']

                fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
                pretrained_model.fc = fc

                from arch.resnet_arch import ResNet, Bottleneck
                from collections import namedtuple
                ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

                if cfg['network_D']['resnet_arch'] == 'resnet50':
                    from arch.resnet_arch import resnet50
                    resnet50_config = ResNetConfig(
                        block=Bottleneck, 
                        n_blocks=[3, 4, 6, 3],
                        channels=[64, 128, 256, 512])
                    self.netD = ResNet(resnet50_config, OUTPUT_DIM)
                elif cfg['network_D']['resnet_arch'] == 'resnet101':
                    from arch.resnet_arch import resnet101
                    resnet101_config = ResNetConfig(
                        block=Bottleneck, 
                        n_blocks=[3, 4, 23, 3],
                        channels=[64, 128, 256, 512])
                    self.netD = ResNet(resnet101_config, OUTPUT_DIM)
                elif cfg['network_D']['resnet_arch'] == 'resnet152':
                    from arch.resnet_arch import resnet152
                    resnet152_config = ResNetConfig(
                        block=Bottleneck,
                        n_blocks=[3, 8, 36, 3],
                        channels=[64, 128, 256, 512])
                    self.netD = ResNet(resnet152_config, OUTPUT_DIM)

                self.netD.load_state_dict(pretrained_model.state_dict())
                print("Resnet pretrain loaded.")

        # ResNeSt
        # ["resnest50", "resnest101", "resnest200", "resnest269"]
        elif cfg['network_D']['netD'] == 'ResNeSt':
            if cfg['network_D']['ResNeSt_pretrain'] == 'resnest50':
                from arch.discriminators import resnest50
                self.netD = resnest50(pretrained=cfg['network_D']['pretrained'],
                                      num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['ResNeSt_pretrain'] == 'resnest101':
                from arch.discriminators import resnest101
                self.netD = resnest101(pretrained=cfg['network_D']['pretrained'],
                                       num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['ResNeSt_pretrain'] == 'resnest200':
                from arch.discriminators import resnest200
                self.netD = resnest200(pretrained=cfg['network_D']['pretrained'],
                                       num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['ResNeSt_pretrain'] == 'resnest269':
                from arch.discriminators import resnest269
                self.netD = resnest269(pretrained=cfg['network_D']['pretrained'],
                                       num_classes=cfg['network_D']['num_classes'])

        # TODO: need fixing
        # FileNotFoundError: [Errno 2] No such file or directory:
        #   '../experiments/pretrained_models/VGG16minc_53.pth'
        # self.netD = MINCFeatureExtractor(
        #   feature_layer=34, use_bn=False, use_input_norm=True, device=torch.device('cpu'))

        # Transformer (Warning: uses own init!)
        elif cfg['network_D']['netD'] == 'TranformerDiscriminator':
            from arch.discriminators import TranformerDiscriminator
            self.netD = TranformerDiscriminator(
                img_size=cfg['network_D']['img_size'],
                patch_size=cfg['network_D']['patch_size'],
                in_chans=cfg['network_D']['in_chans'],
                num_classes=cfg['network_D']['num_classes'],
                embed_dim=cfg['network_D']['embed_dim'],
                depth=cfg['network_D']['depth'],
                num_heads=cfg['network_D']['num_heads'],
                mlp_ratio=cfg['network_D']['mlp_ratio'],
                qkv_bias=cfg['network_D']['qkv_bias'],
                qk_scale=cfg['network_D']['qk_scale'],
                drop_rate=cfg['network_D']['drop_rate'],
                attn_drop_rate=cfg['network_D']['attn_drop_rate'],
                drop_path_rate=cfg['network_D']['drop_path_rate'],
                hybrid_backbone=cfg['network_D']['hybrid_backbone'],
                norm_layer=cfg['network_D']['norm_layer'])

        #############################################

        elif cfg['network_D']['netD'] == 'ViT':
            from vit_pytorch import ViT
            self.netD = ViT(
                image_size=cfg['network_D']['image_size'],
                patch_size=cfg['network_D']['patch_size'],
                num_classes=cfg['network_D']['num_classes'],
                dim=cfg['network_D']['dim'],
                depth=cfg['network_D']['depth'],
                heads=cfg['network_D']['heads'],
                mlp_dim=cfg['network_D']['mlp_dim'],
                dropout=cfg['network_D']['dropout'],
                emb_dropout=cfg['network_D']['emb_dropout']
            )

        elif cfg['network_D']['netD'] == 'DeepViT':
            from vit_pytorch.deepvit import DeepViT
            self.netD = DeepViT(
                image_size=cfg['network_D']['image_size'],
                patch_size=cfg['network_D']['patch_size'],
                num_classes=cfg['network_D']['num_classes'],
                dim=cfg['network_D']['dim'],
                depth=cfg['network_D']['depth'],
                heads=cfg['network_D']['heads'],
                mlp_dim=cfg['network_D']['mlp_dim'],
                dropout=cfg['network_D']['dropout'],
                emb_dropout=cfg['network_D']['emb_dropout']
            )

        #############################################
        # RepVGG-A0, RepVGG-A1, RepVGG-A2, RepVGG-B0, RepVGG-B1, RepVGG-B1g2, RepVGG-B1g4,
        # RepVGG-B2, RepVGG-B2g2, RepVGG-B2g4, RepVGG-B3, RepVGG-B3g2, RepVGG-B3g4
        elif cfg['network_D']['netD'] == 'RepVGG':
            if cfg['network_D']['RepVGG_arch'] == 'RepVGG-A0':
                from arch.RepVGG_arch import create_RepVGG_A0
                self.netD = create_RepVGG_A0(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-A1':
                from arch.RepVGG_arch import create_RepVGG_A1
                self.netD = create_RepVGG_A1(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-A2':
                from arch.RepVGG_arch import create_RepVGG_A2
                self.netD = create_RepVGG_A2(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-B0':
                from arch.RepVGG_arch import create_RepVGG_B0
                self.netD = create_RepVGG_B0(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-B1':
                from arch.RepVGG_arch import create_RepVGG_B1
                self.netD = create_RepVGG_B1(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-B1g2':
                from arch.RepVGG_arch import create_RepVGG_B1g2
                self.netD = create_RepVGG_B1g2(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-B1g4':
                from arch.RepVGG_arch import create_RepVGG_B1g4
                self.netD = create_RepVGG_B1g4(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-B2':
                from arch.RepVGG_arch import create_RepVGG_B2
                self.netD = create_RepVGG_B2(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-B2g2':
                from arch.RepVGG_arch import create_RepVGG_B2g2
                self.netD = create_RepVGG_B2g2(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-B2g4':
                from arch.RepVGG_arch import create_RepVGG_B2g4
                self.netD = create_RepVGG_B2g4(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-B3':
                from arch.RepVGG_arch import create_RepVGG_B3
                self.netD = create_RepVGG_B3(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-B3g2':
                from arch.RepVGG_arch import create_RepVGG_B3g2
                self.netD = create_RepVGG_B3g2(deploy=False, num_classes=cfg['network_D']['num_classes'])
            elif cfg['network_D']['RepVGG_arch'] == 'RepVGG-B3g4':
                from arch.RepVGG_arch import create_RepVGG_B3g4
                self.netD = create_RepVGG_B3g4(deploy=False, num_classes=cfg['network_D']['num_classes'])

        #############################################

        elif cfg['network_D']['netD'] == 'squeezenet':
            from arch.squeezenet_arch import SqueezeNet
            self.netD = SqueezeNet(num_classes=cfg['network_D']['num_classes'], version=cfg['network_D']['version'])

        #############################################

        elif cfg['network_D']['netD'] == 'SwinTransformer':
            from swin_transformer_pytorch import SwinTransformer

            self.netD = SwinTransformer(
                hidden_dim=cfg['network_D']['hidden_dim'],
                layers=cfg['network_D']['layers'],
                heads=cfg['network_D']['heads'],
                channels=cfg['network_D']['channels'],
                num_classes=cfg['network_D']['num_classes'],
                head_dim=cfg['network_D']['head_dim'],
                window_size=cfg['network_D']['window_size'],
                downscaling_factors=cfg['network_D']['downscaling_factors'],
                relative_pos_embedding=cfg['network_D']['relative_pos_embedding'])

        # NFNet
        elif cfg['network_D']['netD'] == 'NFNet':
            from arch.NFNet_arch import NFNet
            self.netD = NFNet(
                num_classes=cfg['network_D']['num_classes'],
                variant=cfg['network_D']['variant'],
                stochdepth_rate=cfg['network_D']['stochdepth_rate'],
                alpha=cfg['network_D']['alpha'],
                se_ratio=cfg['network_D']['se_ratio'],
                activation=cfg['network_D']['activation'])
        elif cfg['network_D']['netD'] == 'lvvit':
            from arch.lvvit_arch import LV_ViT
            self.netD = LV_ViT(
                img_size=cfg['network_D']['img_size'], 
                patch_size=cfg['network_D']['patch_size'], 
                in_chans=cfg['network_D']['in_chans'], 
                num_classes=cfg['network_D']['num_classes'], 
                embed_dim=cfg['network_D']['embed_dim'], 
                depth=cfg['network_D']['depth'],
                num_heads=cfg['network_D']['num_heads'], 
                mlp_ratio=cfg['network_D']['mlp_ratio'], 
                qkv_bias=cfg['network_D']['qkv_bias'], 
                qk_scale=cfg['network_D']['qk_scale'], 
                drop_rate=cfg['network_D']['drop_rate'], 
                attn_drop_rate=cfg['network_D']['attn_drop_rate'],
                drop_path_rate=cfg['network_D']['drop_path_rate'], 
                drop_path_decay=cfg['network_D']['drop_path_decay'], 
                hybrid_backbone=cfg['network_D']['hybrid_backbone'], 
                norm_layer=nn.LayerNorm, 
                p_emb=cfg['network_D']['p_emb'], 
                head_dim=cfg['network_D']['head_dim'],
                skip_lam=cfg['network_D']['skip_lam'],
                order=cfg['network_D']['order'], 
                mix_token=cfg['network_D']['mix_token'], 
                return_dense=cfg['network_D']['return_dense'])
        elif cfg['network_D']['netD'] == 'timm':
            import timm
            self.netD = timm.create_model(cfg['network_D']['timm_model'], 
                                          num_classes=1, pretrained=True)
        elif cfg['network_D']['netD'] == 'resnet3d':
            from arch.resnet3d_arch import generate_model
            self.netD = generate_model(cfg['network_D']['model_depth'])
        elif cfg['network_D']['netD'] == 'FFCNLayerDiscriminator':
            from arch.lama_arch import FFCNLayerDiscriminator
            self.netD = FFCNLayerDiscriminator(3)
        elif cfg['network_D']['netD'] == 'effV2':
            if cfg['network_D']['size'] == "s":
                from arch.efficientnetV2_arch import effnetv2_s
                self.netD = effnetv2_s()
            elif cfg['network_D']['size'] == "m":
                from arch.efficientnetV2_arch import effnetv2_m
                self.netD = effnetv2_m()
            elif cfg['network_D']['size'] == "l":
                from arch.efficientnetV2_arch import effnetv2_l
                self.netD = effnetv2_l()
            elif cfg['network_D']['size'] == "xl":
                from arch.efficientnetV2_arch import effnetv2_xl
                self.netD = effnetv2_xl()
        elif cfg['network_D']['netD'] == 'x_transformers':
            from x_transformers import ViTransformerWrapper, Encoder
            self.netD = ViTransformerWrapper(
                image_size=cfg['network_D']['image_size'],
                patch_size=cfg['network_D']['patch_size'],
                num_classes=1,
                attn_layers=Encoder(
                    dim=cfg['network_D']['dim'],
                    depth=cfg['network_D']['depth'],
                    heads=cfg['network_D']['heads'],
                )
            )
        elif cfg['network_D']['netD'] == 'mobilevit':
            if cfg['network_D']['size'] == "xxs":
                from arch.mobilevit_arch import mobilevit_xxs
                self.netD = mobilevit_xxs()
            elif cfg['network_D']['size'] == "xs":
                from arch.mobilevit_arch import mobilevit_xs
                self.netD = mobilevit_xs()
            elif cfg['network_D']['size'] == "x":
                from arch.mobilevit_arch import mobilevit_s
                self.netD = mobilevit_s()
        elif cfg['network_D']['netD'] == 'hrt':
            from arch.hrt_arch import HighResolutionTransformer
            self.netD = HighResolutionTransformer()

        if cfg['network_D']['WSConv_replace'] == 'True':
            from nfnets import replace_conv, WSConv2d, ScaledStdConv2d
            replace_conv(self.netD, ScaledStdConv2d)

        # only doing init, if not 'TranformerDiscriminator', 'EfficientNet',
        # 'ResNeSt', 'resnet', 'ViT', 'DeepViT', 'mobilenetV3'
        # should probably be rewritten
        if cfg['network_D']['netD'] in \
                ('resnet3d', 'NFNet', 'context_encoder', 'VGG', 'VGG_fea', 'Discriminator_VGG_128_SN',
                 'VGGFeatureExtractor', 'NLayerDiscriminator', 'MultiscaleDiscriminator',
                 'Discriminator_ResNet_128', 'ResNet101FeatureExtractor', 'MINCNet',
                 'PixelDiscriminator', 'ResNeSt', 'RepVGG', 'squeezenet', 'SwinTransformer'):
            if self.global_step == 0:
                weights_init(self.netD, 'kaiming')
                print("Discriminator weight init complete.")

        # loss functions
        self.l1 = nn.L1Loss()

        if cfg['train']['loss_f'] == 'L1Loss':
            loss_f = torch.nn.L1Loss()
        elif cfg['train']['loss_f'] == 'L1CosineSim':
            loss_f = L1CosineSim(loss_lambda=cfg['train']['loss_lambda'],
                                 reduction=cfg['train']['reduction_L1CosineSim'])

        self.HFENLoss = HFENLoss(
            loss_f=loss_f,
            kernel=cfg['train']['kernel'],
            kernel_size=cfg['train']['kernel_size'],
            sigma=cfg['train']['sigma'],
            norm=cfg['train']['norm'])
        self.ElasticLoss = ElasticLoss(
            a=cfg['train']['a'],
            reduction=cfg['train']['reduction_elastic'])
        self.RelativeL1 = RelativeL1(
            eps=cfg['train']['l1_eps'],
            reduction=cfg['train']['reduction_relative'])
        self.L1CosineSim = L1CosineSim(
            loss_lambda=cfg['train']['loss_lambda'],
            reduction=cfg['train']['reduction_L1CosineSim'])
        self.ClipL1 = ClipL1(
            clip_min=cfg['train']['clip_min'],
            clip_max=cfg['train']['clip_max'])

        if cfg['train']['loss_f_fft'] == 'L1Loss':
            loss_f_fft = torch.nn.L1Loss
        elif cfg['train']['loss_f_fft'] == 'L1CosineSim':
            loss_f_fft = L1CosineSim(
                loss_lambda=cfg['train']['loss_lambda'], 
                reduction=cfg['train']['reduction_L1CosineSim'])

        self.FFTloss = FFTloss(
            loss_f=loss_f_fft, 
            reduction=cfg['train']['reduction_fft'])
        self.OFLoss = OFLoss()
        self.GPLoss = GPLoss(
            trace=cfg['train']['gp_trace'], 
            spl_denorm=cfg['train']['gp_spl_denorm'])
        self.CPLoss = CPLoss(
            rgb=cfg['train']['rgb'], 
            yuv=cfg['train']['yuv'], 
            yuvgrad=cfg['train']['yuvgrad'], 
            trace=cfg['train']['cp_trace'], 
            spl_denorm=cfg['train']['cp_spl_denorm'], 
            yuv_denorm=cfg['train']['yuv_denorm'])
        self.StyleLoss = StyleLoss()
        self.TVLoss = TVLoss(
            tv_type=cfg['train']['tv_type'], 
            p=cfg['train']['p'])
        self.Contextual_Loss = Contextual_Loss(
            cfg['train']['layers_weights'], 
            crop_quarter=cfg['train']['crop_quarter'], 
            max_1d_size=cfg['train']['max_1d_size'],
            distance_type=cfg['train']['distance_type'], 
            b=cfg['train']['b'], 
            band_width=cfg['train']['band_width'],
            use_vgg=cfg['train']['use_vgg'], 
            net=cfg['train']['net_contextual'], 
            calc_type=cfg['train']['calc_type'], 
            use_timm=cfg['train']['use_timm'], 
            timm_model=cfg['train']['timm_model'])

        self.MSELoss = torch.nn.MSELoss()
        self.L1Loss = nn.L1Loss()
        self.BCELogits = torch.nn.BCEWithLogitsLoss()
        self.BCE = torch.nn.BCELoss()
        self.FFLoss = FocalFrequencyLoss()

        # perceptual loss
        from arch.networks_basic import PNetLin
        self.perceptual_loss = PNetLin(
            pnet_rand=cfg['train']['pnet_rand'],
            pnet_tune=cfg['train']['pnet_tune'],
            pnet_type=cfg['train']['pnet_type'],
            use_dropout=cfg['train']['use_dropout'],
            spatial=cfg['train']['spatial'],
            version=cfg['train']['version'],
            lpips=cfg['train']['lpips'])
        model_path = os.path.abspath(f'loss/lpips_weights/v0.1/{cfg["train"]["pnet_type"]}.pth')
        print(f'Loading model from: {model_path}')
        self.perceptual_loss.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device)), strict=False)
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False

        if cfg['train']['force_fp16_perceptual'] is True and cfg['train']['perceptual_tensorrt'] is False:
            print("Converting perceptual model to FP16")
            self.perceptual_loss = self.perceptual_loss.half()

        if cfg['train']['force_fp16_perceptual'] is True and cfg['train']['perceptual_tensorrt'] is True:
            print("Converting perceptual model to TensorRT (FP16)")
            import torch_tensorrt
            example_data = torch.rand(1, 3, 256, 448).half().cuda()
            self.perceptual_loss = self.perceptual_loss.half().cuda()
            self.perceptual_loss = torch.jit.trace(self.perceptual_loss, [example_data, example_data])
            self.perceptual_loss = torch_tensorrt.compile(
                self.perceptual_loss,
                inputs=[torch_tensorrt.Input(
                            min_shape=(1, 3, 64, 64), opt_shape=(1, 3, 256, 448),
                            max_shape=(1, 3, 720, 1280), dtype=torch.half),
                        torch_tensorrt.Input(
                            min_shape=(1, 3, 64, 64), opt_shape=(1, 3, 256, 448),
                            max_shape=(1, 3, 720, 1280), dtype=torch.half)],
                enabled_precisions={torch.half}, truncate_long_and_double=True)
            del example_data

        elif cfg['train']['force_fp16_perceptual'] is False and cfg['train']['perceptual_tensorrt'] is True:
            print("Converting perceptual model to TensorRT")
            import torch_tensorrt
            example_data = torch.rand(1, 3, 256, 448)
            self.perceptual_loss = torch.jit.trace(
                self.perceptual_loss, [example_data, example_data])
            self.perceptual_loss = torch_tensorrt.compile(
                self.perceptual_loss,
                inputs=[torch_tensorrt.Input(
                            min_shape=(1, 3, 64, 64), opt_shape=(1, 3, 256, 448),
                            max_shape=(1, 3, 720, 1280), dtype=torch.float32),
                        torch_tensorrt.Input(
                            min_shape=(1, 3, 64, 64), opt_shape=(1, 3, 256, 448),
                            max_shape=(1, 3, 720, 1280), dtype=torch.float32)],
                enabled_precisions={torch.float},
                truncate_long_and_double=True)
            del example_data

        from arch.hrf_perceptual import ResNetPL
        self.hrf_perceptual_loss = ResNetPL()
        for param in self.hrf_perceptual_loss.parameters():
            param.requires_grad = False

        if cfg['train']['force_fp16_hrf'] is True:
            self.hrf_perceptual_loss = self.hrf_perceptual_loss.half()

        self.ColorLoss = ColorLoss()
        self.FrobeniusNormLoss = FrobeniusNormLoss()
        self.GradientLoss = GradientLoss()
        self.MultiscalePixelLoss = MultiscalePixelLoss()
        self.SPLoss = SPLoss()

        # pytorch loss
        self.HuberLoss = nn.HuberLoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.SoftMarginLoss = nn.SoftMarginLoss()

        self.LapLoss = LapLoss()

        # piq loss
        self.SSIMLoss = SSIMLoss()
        self.MultiScaleSSIMLoss = MultiScaleSSIMLoss()
        self.VIFLoss = VIFLoss()
        self.FSIMLoss = FSIMLoss()
        self.GMSDLoss = GMSDLoss()
        self.MultiScaleGMSDLoss = MultiScaleGMSDLoss()
        self.VSILoss = VSILoss()
        self.HaarPSILoss = HaarPSILoss()
        self.MDSILoss = MDSILoss()
        self.BRISQUELoss = BRISQUELoss()
        self.PieAPP = PieAPP(enable_grad=True)
        self.DISTS = DISTS()
        self.IS = IS()
        self.FID = FID()
        self.KID = KID()
        self.PR = PR()

        if cfg['network_G']['netG'] == 'rife':
            from loss.loss import SOBEL
            self.sobel = SOBEL()

        # discriminator loss
        if cfg['network_D']['discriminator_criterion'] == "MSE":
            self.discriminator_criterion = torch.nn.MSELoss()

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

        # augmentation
        if cfg['train']['augmentation_method'] == "MuarAugment":
            from loss.MuarAugment import BatchRandAugment, MuAugment
            rand_augment = BatchRandAugment(
                N_TFMS=cfg['train']['N_TFMS'], 
                MAGN=cfg['train']['MAGN'], 
                mean=[0.7032, 0.6346, 0.6234], 
                std=[0.2520, 0.2507, 0.2417])
            self.mu_transform = MuAugment(
                rand_augment, 
                N_COMPS=cfg['train']['N_COMPS'], 
                N_SELECTED=cfg['train']['N_SELECTED'])
        elif cfg['train']['augmentation_method'] == "batch_aug":
            from loss.batchaug import BatchAugment
            self.batch_aug = BatchAugment(
                mixopts=cfg['train']['mixopts'], 
                mixprob=cfg['train']['mixprob'], 
                mixalpha=cfg['train']['mixalpha'], 
                aux_mixprob=cfg['train']['aux_mixprob'], 
                aux_mixalpha=cfg['train']['aux_mixalpha'])

        """if cfg['network_G']['CEM'] is True:
            from arch.CEM import CEMnet  # import is unused"""

    def forward(self, image, masks):
        return self.netG(image, masks)

    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        # iteration count is sometimes broken, adding a check and manual increment
        # only increment if generator gets trained (loop gets called a second time for discriminator)
        if self.trainer.global_step != 0:
            if optimizer_idx == 0 and self.iter_check == self.trainer.global_step:
                self.trainer.global_step += 1
            self.iter_check = self.trainer.global_step

        # inpainting:
        # train_batch[0][0] = batch_size
        # train_batch[0] = masked
        # train_batch[1] = mask (lr)
        # train_batch[2] = original
        # train_batch[3] = edge
        # train_batch[4] = grayscale

        # super resolution
        # train_batch[0] = Null
        # train_batch[1] = lr
        # train_batch[2] = hr
        # train_batch[3] = landmarks (DFDNet)

        # frame interpolation
        # train_batch[0] = 1st frame
        # train_batch[1] = 3rd frame
        # train_batch[2] = 2nd frame, gets generated

        # train generator
        ############################
        if cfg['network_G']['netG'] == 'CTSDG':
            # input_image, input_edge, mask
            out, projected_image, projected_edge = self.netG(train_batch[0], 
                                                             train_batch[3], 
                                                             train_batch[1])

        if cfg['network_G']['netG'] in \
                ('lama', 'MST', 'MANet', 'context_encoder', 'DFNet', 'AdaFill', 'MEDFE',
                 'RFR', 'LBAM', 'DMFN', 'Partial', 'RN', 'RN', 'DSNet', 'DSNetRRDB',
                 'DSNetDeoldify'):
            # generate fake (1 output)
            out = self(train_batch[0], train_batch[1])

            # masking, taking original content from HR
            out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

        ############################
        if (cfg['network_G']['netG'] == 'deepfillv1' 
                or cfg['network_G']['netG'] == 'deepfillv2' 
                or cfg['network_G']['netG'] == 'Adaptive'):
            # generate fake (2 outputs)
            out, other_img = self(train_batch[0], train_batch[1])

            # masking, taking original content from HR
            out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

        ############################
        # exotic generators
        # CSA
        if cfg['network_G']['netG'] == 'CSA':
            coarse_result, out, csa, csa_d = self(train_batch[0], train_batch[1])
            # masking, taking original content from HR
            out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

        # EdgeConnect
        # train_batch[3] = edges
        # train_batch[4] = grayscale
        if cfg['network_G']['netG'] == 'EdgeConnect':
            out, other_img = self.netG(train_batch[0], train_batch[3], train_batch[4], train_batch[1])
            # masking, taking original content from HR
            out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

        # PVRS
        if cfg['network_G']['netG'] == 'PVRS':
            out, _, edge_small, edge_big = self.netG(train_batch[0], train_batch[1], train_batch[3])
            # masking, taking original content from HR
            out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

        # FRRN
        if cfg['network_G']['netG'] == 'FRRN':
            out, mid_x, mid_mask = self(train_batch[0], train_batch[1])

            # masking, taking original content from HR
            out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

        # deoldify
        if cfg['network_G']['netG'] == 'deoldify':
            out = self.netG(train_batch[0])

        ############################
        # if frame interpolation
        if cfg['network_G']['netG'] in ("CDFI", "sepconv_enhanced", 'CAIN', 'RRIN', 'ABME', 'EDSC'):
            out = self.netG(train_batch[0], train_batch[1])

        if cfg['network_G']['netG'] == 'rife':
            out, flow = self.netG(train_batch[0], train_batch[1], training=True)

        # ESRT / swinir / lightweight_gan / RRDB_net / GLEAN / GPEN / comodgan
        if cfg['network_G']['netG'] in \
                ("restormer", "SRVGGNetCompact", "ESRT", "swinir", 'lightweight_gan', 'RRDB_net',
                 'GLEAN', 'GPEN', 'comodgan'):
            if (cfg['datasets']['train']['mode'] == 'DS_inpaint'
                    or cfg['datasets']['train']['mode'] == 'DS_inpaint_tiled'
                    or cfg['datasets']['train']['mode'] == 'DS_inpaint_tiled_batch'):
                # masked test with inpaint dataloader
                tmp = torch.cat([train_batch[0], train_batch[1]], 1)
                out = self.netG(tmp)
                out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])
            else:
                # normal dataloader
                out = self.netG(train_batch[1])
                # # unpad images if using CEM
                if cfg['network_G']['CEM'] is True:
                    out = self.CEM_net.HR_unpadder(out)
                    train_batch[2] = self.CEM_net.HR_unpadder(train_batch[2])

        # GFPGAN
        if cfg['network_G']['netG'] == 'GFPGAN':
            if (cfg['datasets']['train']['mode'] == 'DS_inpaint'
                    or cfg['datasets']['train']['mode'] == 'DS_inpaint_tiled'
                    or cfg['datasets']['train']['mode'] == 'DS_inpaint_tiled_batch'):
                # masked test with inpaint dataloader
                tmp = torch.cat([train_batch[0], train_batch[1]], 1)
                out, _ = self.netG(tmp)
                out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])
            else:
                out, _ = self.netG(train_batch[1])

        if cfg['network_G']['netG'] == 'srflow':
            # freeze rrdb in the beginning
            if self.trainer.global_step < cfg['network_G']['freeze_iter']:
                self.netG.set_rrdb_training(False)
            else:
                self.netG.set_rrdb_training(True)
            z, nll, y_logits = self.netG(gt=train_batch[2], lr=train_batch[1], reverse=False)
            out, logdet = self.netG(lr=train_batch[1], z=z, eps_std=0, reverse=True, reverse_with_grad=True)
            # out = torch.clamp(out, 0, 1)  # forcing out to be between 0 and 1

        # DFDNet
        if cfg['network_G']['netG'] == 'DFDNet':
            out = self.netG(train_batch[1], part_locations=train_batch[3])
            # range [-1, 1] to [0, 1]
            out = out + 1
            out = out - out.min()
            out = out / (out.max() - out.min())

        # train generator
        if optimizer_idx == 0:
            ############################
            # loss calculation
            total_loss = 0
            if cfg['train']['L1Loss_weight'] > 0:
                L1Loss_forward = cfg['train']['L1Loss_weight']*self.L1Loss(out, train_batch[2])
                total_loss += L1Loss_forward
                writer.add_scalar('loss/L1', L1Loss_forward, self.trainer.global_step)

            if cfg['train']['HFEN_weight'] > 0:
                HFENLoss_forward = cfg['train']['HFEN_weight']*self.HFENLoss(out, train_batch[2])
                total_loss += HFENLoss_forward
                writer.add_scalar('loss/HFEN', HFENLoss_forward, self.trainer.global_step)

            if cfg['train']['Elastic_weight'] > 0:
                ElasticLoss_forward = cfg['train']['Elastic_weight']*self.ElasticLoss(out, train_batch[2])
                total_loss += ElasticLoss_forward
                writer.add_scalar('loss/Elastic', ElasticLoss_forward, self.trainer.global_step)

            if cfg['train']['Relative_l1_weight'] > 0:
                RelativeL1_forward = cfg['train']['Relative_l1_weight']*self.RelativeL1(out, train_batch[2])
                total_loss += RelativeL1_forward
                writer.add_scalar('loss/RelativeL1', RelativeL1_forward, self.trainer.global_step)

            if cfg['train']['L1CosineSim_weight'] > 0:
                L1CosineSim_forward = cfg['train']['L1CosineSim_weight']*self.L1CosineSim(out, train_batch[2])
                total_loss += L1CosineSim_forward
                writer.add_scalar('loss/L1CosineSim', L1CosineSim_forward, self.trainer.global_step)

            if cfg['train']['ClipL1_weight'] > 0:
                ClipL1_forward = cfg['train']['ClipL1_weight']*self.ClipL1(out, train_batch[2])
                total_loss += ClipL1_forward
                writer.add_scalar('loss/ClipL1', ClipL1_forward, self.trainer.global_step)

            if cfg['train']['FFTLoss_weight'] > 0:
                FFTloss_forward = cfg['train']['FFTLoss_weight']*self.FFTloss(out, train_batch[2])
                total_loss += FFTloss_forward
                writer.add_scalar('loss/FFT', FFTloss_forward, self.trainer.global_step)

            if cfg['train']['OFLoss_weight'] > 0:
                OFLoss_forward = cfg['train']['OFLoss_weight']*self.OFLoss(out)
                total_loss += OFLoss_forward
                writer.add_scalar('loss/OF', OFLoss_forward, self.trainer.global_step)

            if cfg['train']['GPLoss_weight'] > 0:
                GPLoss_forward = cfg['train']['GPLoss_weight']*self.GPLoss(out, train_batch[2])
                total_loss += GPLoss_forward
                writer.add_scalar('loss/GP', GPLoss_forward, self.trainer.global_step)

            if cfg['train']['CPLoss_weight'] > 0:
                CPLoss_forward = cfg['train']['CPLoss_weight']*self.CPLoss(out, train_batch[2])
                total_loss += CPLoss_forward
                writer.add_scalar('loss/CP', CPLoss_forward, self.trainer.global_step)

            if cfg['train']['Contextual_weight'] > 0:
                Contextual_Loss_forward = cfg['train']['Contextual_weight']*self.Contextual_Loss(out, train_batch[2])
                total_loss += Contextual_Loss_forward
                writer.add_scalar('loss/contextual', Contextual_Loss_forward, self.trainer.global_step)

            if cfg['train']['StyleLoss_weight'] > 0:
                style_forward = cfg['train']['StyleLoss_weight']*self.StyleLoss(out, train_batch[2])
                total_loss += style_forward
                writer.add_scalar('loss/style', style_forward, self.trainer.global_step)

            if cfg['train']['TVLoss_weight'] > 0:
                tv_forward = cfg['train']['TVLoss_weight']*self.TVLoss(out)
                total_loss += tv_forward
                writer.add_scalar('loss/tv', tv_forward, self.trainer.global_step)

            if cfg['train']['perceptual_weight'] > 0:
                self.perceptual_loss.to(self.device)
                if (cfg['train']['force_fp16_perceptual'] is True
                        and cfg['train']['perceptual_tensorrt'] is True):
                    perceptual_loss_forward = \
                        cfg['train']['perceptual_weight'] * \
                        self.perceptual_loss(out.half(), train_batch[2].half())[0]
                elif (cfg['train']['force_fp16_perceptual'] is True
                      and cfg['train']['perceptual_tensorrt'] is False):
                    perceptual_loss_forward = \
                        cfg['train']['perceptual_weight'] * \
                        self.perceptual_loss(in0=out.half(), in1=train_batch[2].half())
                elif (cfg['train']['force_fp16_perceptual'] is False
                      and cfg['train']['perceptual_tensorrt'] is True):
                    perceptual_loss_forward = \
                        cfg['train']['perceptual_weight'] * \
                        self.perceptual_loss(out, train_batch[2])[0]
                elif cfg['train']['force_fp16_perceptual'] is False and cfg['train']['perceptual_tensorrt'] is False:
                    perceptual_loss_forward = \
                        cfg['train']['perceptual_weight'] * \
                        self.perceptual_loss(in0=out, in1=train_batch[2])
                writer.add_scalar(
                    'loss/perceptual', perceptual_loss_forward, self.trainer.global_step)
                total_loss += perceptual_loss_forward

            if cfg['train']['hrf_perceptual_weight'] > 0:
                self.hrf_perceptual_loss.to(self.device)
                if cfg['train']['force_fp16_hrf'] is True:
                    hrf_perceptual_loss_forward = \
                        cfg['train']['hrf_perceptual_weight'] * \
                        self.hrf_perceptual_loss(out.half(), train_batch[2].half())
                else:
                    hrf_perceptual_loss_forward = \
                        cfg['train']['hrf_perceptual_weight'] * \
                        self.hrf_perceptual_loss(out, train_batch[2])
                writer.add_scalar(
                    'loss/hrf_perceptual', hrf_perceptual_loss_forward, self.trainer.global_step)
                total_loss += hrf_perceptual_loss_forward

            if cfg['train']['MSE_weight'] > 0:
                MSE_forward = cfg['train']['MSE_weight']*self.MSELoss(out, train_batch[2])
                total_loss += MSE_forward
                writer.add_scalar('loss/MSE', MSE_forward, self.trainer.global_step)

            if cfg['train']['BCE_weight'] > 0:
                BCELogits_forward = cfg['train']['BCE_weight']*self.BCELogits(out, train_batch[2])
                total_loss += BCELogits_forward
                writer.add_scalar('loss/BCELogits', BCELogits_forward, self.trainer.global_step)

            if cfg['train']['Huber_weight'] > 0:
                Huber_forward = cfg['train']['Huber_weight']*self.HuberLoss(out, train_batch[2])
                total_loss += Huber_forward
                writer.add_scalar('loss/Huber', Huber_forward, self.trainer.global_step)

            if cfg['train']['SmoothL1_weight'] > 0:
                SmoothL1_forward = cfg['train']['SmoothL1_weight']*self.SmoothL1Loss(out, train_batch[2])
                total_loss += SmoothL1_forward
                writer.add_scalar('loss/SmoothL1', SmoothL1_forward, self.trainer.global_step)

            if cfg['train']['Lap_weight'] > 0:
                Lap_forward = cfg['train']['Lap_weight']*(self.LapLoss(out, train_batch[2])).mean()
                total_loss += Lap_forward
                writer.add_scalar('loss/Lap', Lap_forward, self.trainer.global_step)

            if cfg['train']['ColorLoss_weight'] > 0:
                ColorLoss_forward = cfg['train']['ColorLoss_weight']*(self.ColorLoss(out, train_batch[2]))
                total_loss += ColorLoss_forward
                writer.add_scalar('loss/ColorLoss', ColorLoss_forward, self.trainer.global_step)

            if cfg['train']['FrobeniusNormLoss_weight'] > 0:
                FrobeniusNormLoss_forward = \
                    cfg['train']['FrobeniusNormLoss_weight'] * \
                    self.FrobeniusNormLoss(out, train_batch[2])
                total_loss += FrobeniusNormLoss_forward
                writer.add_scalar(
                    'loss/FrobeniusNormLoss', FrobeniusNormLoss_forward, self.trainer.global_step)

            if cfg['train']['GradientLoss_weight'] > 0:
                GradientLoss_forward = \
                    cfg['train']['GradientLoss_weight'] * \
                    self.GradientLoss(out, train_batch[2])
                total_loss += GradientLoss_forward
                writer.add_scalar(
                    'loss/GradientLoss', GradientLoss_forward, self.trainer.global_step)

            if cfg['train']['MultiscalePixelLoss_weight'] > 0:
                MultiscalePixelLoss_forward = \
                    cfg['train']['MultiscalePixelLoss_weight'] * \
                    self.MultiscalePixelLoss(out, train_batch[2])
                total_loss += MultiscalePixelLoss_forward
                writer.add_scalar(
                    'loss/MultiscalePixelLoss', MultiscalePixelLoss_forward, self.trainer.global_step)

            if cfg['train']['SPLoss_weight'] > 0:
                SPLoss_forward = cfg['train']['SPLoss_weight'] * (self.SPLoss(out, train_batch[2]))
                total_loss += SPLoss_forward
                writer.add_scalar('loss/SPLoss', SPLoss_forward, self.trainer.global_step)

            if cfg['train']['FFLoss_weight'] > 0:
                FFLoss_forward = \
                    cfg['train']['FFLoss_weight'] * \
                    self.FFLoss(out.type(torch.cuda.FloatTensor),
                                train_batch[2].type(torch.cuda.FloatTensor))
                total_loss += FFLoss_forward
                writer.add_scalar('loss/FFLoss', FFLoss_forward, self.trainer.global_step)

            if cfg['train']['SSIMLoss_weight'] > 0:
                SSIMLoss_forward = cfg['train']['SSIMLoss_weight'] * self.SSIMLoss(out, train_batch[2])
                total_loss += SSIMLoss_forward
                writer.add_scalar('loss/SSIM', SSIMLoss_forward, self.trainer.global_step)

            if cfg['train']['MultiScaleSSIMLoss_weight'] > 0:
                MultiScaleSSIMLoss_forward = \
                    cfg['train']['MultiScaleSSIMLoss_weight'] * \
                    (self.MultiScaleSSIMLoss(out, train_batch[2]))
                total_loss += MultiScaleSSIMLoss_forward
                writer.add_scalar(
                    'loss/MultiScaleSSIM', MultiScaleSSIMLoss_forward, self.trainer.global_step)

            if cfg['train']['VIFLoss_weight'] > 0:
                VIFLoss_forward = cfg['train']['VIFLoss_weight'] * self.VIFLoss(out, train_batch[2])
                total_loss += VIFLoss_forward
                writer.add_scalar('loss/VIF', VIFLoss_forward, self.trainer.global_step)

            if cfg['train']['FSIMLoss_weight'] > 0:
                FSIMLoss_forward = cfg['train']['FSIMLoss_weight'] * self.FSIMLoss(out, train_batch[2])
                total_loss += FSIMLoss_forward
                writer.add_scalar('loss/FSIM', FSIMLoss_forward, self.trainer.global_step)

            if cfg['train']['GMSDLoss_weight'] > 0:
                GMSDLoss_forward = cfg['train']['GMSDLoss_weight'] * self.GMSDLoss(out, train_batch[2])
                total_loss += GMSDLoss_forward
                writer.add_scalar('loss/GMSD', GMSDLoss_forward, self.trainer.global_step)

            if cfg['train']['MultiScaleGMSDLoss_weight'] > 0:
                MultiScaleGMSDLoss_forward = \
                    cfg['train']['MultiScaleGMSDLoss_weight'] * \
                    self.MultiScaleGMSDLoss(out, train_batch[2])
                total_loss += MultiScaleGMSDLoss_forward
                writer.add_scalar(
                    'loss/MultiScaleGMSD', MultiScaleGMSDLoss_forward, self.trainer.global_step)

            if cfg['train']['VSILoss_weight'] > 0:
                VSILoss_forward = cfg['train']['VSILoss_weight']*(self.VSILoss(out, train_batch[2]))
                total_loss += VSILoss_forward
                writer.add_scalar('loss/VSI', VSILoss_forward, self.trainer.global_step)

            if cfg['train']['HaarPSILoss_weight'] > 0:
                HaarPSILoss_forward = \
                    cfg['train']['HaarPSILoss_weight'] * self.HaarPSILoss(out, train_batch[2])
                total_loss += HaarPSILoss_forward
                writer.add_scalar(
                    'loss/HaarPSI', HaarPSILoss_forward, self.trainer.global_step)

            if cfg['train']['MDSILoss_weight'] > 0:
                MDSILoss_forward = cfg['train']['MDSILoss_weight'] * self.MDSILoss(out, train_batch[2])
                total_loss += MDSILoss_forward
                writer.add_scalar('loss/DSI', MDSILoss_forward, self.trainer.global_step)

            if cfg['train']['BRISQUELoss_weight'] > 0:
                BRISQUELoss_forward = cfg['train']['BRISQUELoss_weight'] * self.BRISQUELoss(out)
                total_loss += BRISQUELoss_forward
                writer.add_scalar('loss/BRISQUE', BRISQUELoss_forward, self.trainer.global_step)

            if cfg['train']['PieAPP_weight'] > 0:
                PieAPP_forward = cfg['train']['PieAPP_weight'] * self.PieAPP(out, train_batch[2])
                total_loss += PieAPP_forward
                writer.add_scalar('loss/PieAPP', PieAPP_forward, self.trainer.global_step)

            if cfg['train']['DISTS_weight'] > 0:
                DISTS_forward = cfg['train']['DISTS_weight'] * self.DISTS(out, train_batch[2])
                total_loss += DISTS_forward
                writer.add_scalar('loss/DISTS', DISTS_forward, self.trainer.global_step)

            if cfg['train']['IS_weight'] > 0:
                if cfg['train']['force_piq_fp16'] is True:
                    i1 = self.piq_model(out.half())
                    i2 = self.piq_model(train_batch[2].half())
                else:
                    i1 = self.piq_model(out)
                    i2 = self.piq_model(train_batch[2])
                IS_forward = cfg['train']['IS_weight']*self.IS(i1, i2)
                total_loss += IS_forward
                writer.add_scalar('loss/IS', IS_forward, self.trainer.global_step)

            if cfg['train']['FID_weight'] > 0:
                if cfg['train']['force_piq_fp16'] is True:
                    i1 = self.piq_model(out.half())
                    i2 = self.piq_model(train_batch[2].half())
                else:
                    i1 = self.piq_model(out)
                    i2 = self.piq_model(train_batch[2])
                FID_forward = cfg['train']['FID_weight']*self.FID(i1, i2)
                total_loss += FID_forward
                writer.add_scalar('loss/FID', FID_forward, self.trainer.global_step)

            if cfg['train']['KID_weight'] > 0:
                if cfg['train']['force_piq_fp16'] is True:
                    i1 = self.piq_model(out.half())
                    i2 = self.piq_model(train_batch[2].half())
                else:
                    i1 = self.piq_model(out)
                    i2 = self.piq_model(train_batch[2])
                KID_forward = cfg['train']['KID_weight']*self.KID(i1, i2)
                total_loss += KID_forward
                writer.add_scalar('loss/KID', KID_forward, self.trainer.global_step)

            if cfg['train']['PR_weight'] > 0:
                precision, recall = self.PR(self.piq_model(out), self.piq_model(train_batch[2]))
                PR_forward = cfg['train']['PR_weight']*(precision**-1)
                total_loss += PR_forward
                writer.add_scalar('loss/PR', PR_forward, self.trainer.global_step)

            #########################
            # exotic loss
            # if model has two output, also calculate loss for such an image
            # example with just l1 loss
            if (cfg['network_G']['netG'] == 'deepfillv1' 
                    or cfg['network_G']['netG'] == 'deepfillv2' 
                    or cfg['network_G']['netG'] == 'Adaptive'):
                l1_stage1 = cfg['train']['stage1_weight']*self.L1Loss(other_img, train_batch[2])
                total_loss += l1_stage1
                writer.add_scalar('loss/l1_stage1', l1_stage1, self.trainer.global_step)

            # CSA Loss
            if cfg['network_G']['netG'] == 'CSA':
                recon_loss = self.L1Loss(coarse_result, train_batch[2]) + self.L1Loss(out, train_batch[2])
                cons = ConsistencyLoss()
                cons_loss = cons(csa, csa_d, train_batch[2], train_batch[1])
                writer.add_scalar('loss/recon_loss', recon_loss, self.trainer.global_step)
                total_loss += recon_loss
                writer.add_scalar('loss/cons_loss', cons_loss, self.trainer.global_step)
                total_loss += cons_loss

            # EdgeConnect
            # train_batch[3] = edges
            # train_batch[4] = grayscale
            if cfg['network_G']['netG'] == 'EdgeConnect':
                l1_edge = self.L1Loss(other_img, train_batch[3])
                total_loss += l1_edge
                writer.add_scalar('loss/l1_edge', l1_edge, self.trainer.global_step)

            # PVRS
            if cfg['network_G']['netG'] == 'PVRS':
                edge_big_l1 = self.L1Loss(edge_big, train_batch[3])
                edge_small_l1 = self.L1Loss(
                    edge_small, torch.nn.functional.interpolate(train_batch[3], scale_factor=0.5))
                total_loss += edge_big_l1
                total_loss += edge_small_l1
                writer.add_scalar('loss/edge_big_l1', edge_big_l1, self.trainer.global_step)
                writer.add_scalar('loss/edge_small_l1', edge_small_l1, self.trainer.global_step)

            # FRRN
            if cfg['network_G']['netG'] == 'FRRN':
                mid_l1_loss = 0
                for idx in range(len(mid_x) - 1):
                    mid_l1_loss += self.L1Loss(mid_x[idx] * mid_mask[idx], train_batch[2] * mid_mask[idx])
                total_loss += mid_l1_loss
                writer.add_scalar('loss/mid_l1_loss', mid_l1_loss, self.trainer.global_step)

            writer.add_scalar('loss/g_loss', total_loss, self.trainer.global_step)

            # srflow
            if cfg['network_G']['netG'] == 'srflow':
                nll_loss = torch.mean(nll)
                total_loss += cfg['network_G']['nll_weight']*nll_loss
                writer.add_scalar('loss/nll_loss', nll_loss, self.trainer.global_step)

            # CTSDG
            if cfg['network_G']['netG'] == 'CTSDG':
                edge_loss = self.BCE(projected_edge, train_batch[3]) * cfg['train']['CTSDG_edge_weight']
                total_loss += edge_loss
                writer.add_scalar('loss/edge_loss', edge_loss, self.trainer.global_step)

                projected_loss = self.L1Loss(projected_image, train_batch[2]) * cfg['train']['CTSDG_projected_weight']
                total_loss += projected_loss
                writer.add_scalar('loss/projected_loss', projected_loss, self.trainer.global_step)

            # rife
            if cfg['network_G']['netG'] == 'rife':
                sobel_loss = self.sobel(flow[3], flow[3]*0).mean() * cfg['train']['SOBEL_weight']
                total_loss += sobel_loss
                writer.add_scalar('loss/sobel_loss', sobel_loss, self.trainer.global_step)

            # return total_loss
            #########################
            if cfg['network_D']['netD'] is not None:
                # Try to fool the discriminator
                Tensor = torch.FloatTensor
                fake = Variable(Tensor((out.shape[0])).fill_(0.0), requires_grad=False).unsqueeze(-1).to(self.device)
                if cfg['network_D']['netD'] == 'resnet3d':
                    # 3d
                    if cfg['train']['augmentation_method'] == "diffaug":
                        d_loss_fool = \
                            cfg["network_D"]["d_loss_fool_weight"] * \
                            self.discriminator_criterion(
                                self.netD(DiffAugment(
                                    torch.stack([train_batch[0], out, train_batch[1]], dim=1), 
                                    cfg['train']['policy'])), fake)
                    else:
                        d_loss_fool = \
                            cfg["network_D"]["d_loss_fool_weight"] * \
                            self.discriminator_criterion(
                                self.netD(torch.stack(
                                    [train_batch[0], out, train_batch[1]], dim=1)), fake)
                else:
                    # 2d
                    if cfg['train']['augmentation_method'] == "diffaug":
                        if cfg["network_D"]['netD'] == "MultiscaleDiscriminator":
                          d_loss_fool = 0
                          d_out = self.netD(DiffAugment(out, cfg['train']['policy']))
                          for i in d_out:
                            d_loss_fool += cfg["network_D"]["d_loss_fool_weight"] * \
                                self.discriminator_criterion( \
                                torch.mean(torch.mean(i[0], dim=2), dim=2), fake)
                        else:
                          d_loss_fool = \
                              cfg["network_D"]["d_loss_fool_weight"] * \
                              self.discriminator_criterion(
                                  self.netD(DiffAugment(out, cfg['train']['policy'])), fake)
                    elif cfg['train']['augmentation_method'] == "MuarAugment":
                        self.mu_transform.setup(self)
                        mu_augment, _ = self.mu_transform((out, fake))
                        if cfg["network_D"]['netD'] == "MultiscaleDiscriminator":
                          d_loss_fool = 0
                          d_out = self.netD(mu_augment)
                          for i in d_out:
                            d_loss_fool += cfg["network_D"]["d_loss_fool_weight"] * \
                                self.discriminator_criterion( \
                                torch.mean(torch.mean(i[0], dim=2), dim=2).float(), fake.float())
                        else:
                          d_loss_fool = \
                              cfg["network_D"]["d_loss_fool_weight"] * \
                              self.discriminator_criterion(
                                  self.netD(mu_augment).float(), fake.float())
                    else:
                        if cfg['network_D']['netD'] == 'FFCNLayerDiscriminator':
                            FFCN_class, FFCN_feature = self.netD(out)
                            d_loss_fool = \
                                cfg["network_D"]["d_loss_fool_weight"] * \
                                self.discriminator_criterion(FFCN_class, fake)
                        else:
                            if cfg["network_D"]['netD'] == "MultiscaleDiscriminator":
                              d_loss_fool = 0
                              d_out = self.netD(out)
                              for i in d_out:
                                d_loss_fool += cfg["network_D"]["d_loss_fool_weight"] * \
                                    self.discriminator_criterion( \
                                    torch.mean(torch.mean(i[0], dim=2), dim=2), fake)
                            else:
                              d_loss_fool = \
                                  cfg["network_D"]["d_loss_fool_weight"] * \
                                  self.discriminator_criterion(self.netD(out), fake)

                writer.add_scalar('loss/d_loss_fool', d_loss_fool, self.trainer.global_step)

            if (cfg['network_D']['netD'] == 'FFCNLayerDiscriminator' 
                    and cfg['network_D']['FFCN_feature_weight'] > 0):
                FFCN_class_orig, FFCN_feature_orig = self.netD(train_batch[2])
                # dont give mask if it's not available
                if cfg['network_G']['netG'] in \
                        ("CDFI", "sepconv_enhanced", 'CAIN', 'rife', 'RRIN', 'ABME', 'EDSC'):
                    feature_matching_loss_forward = \
                        cfg['network_D']['FFCN_feature_weight'] * \
                        feature_matching_loss(FFCN_feature, FFCN_feature_orig)
                else:
                    feature_matching_loss_forward = \
                        cfg['network_D']['FFCN_feature_weight'] * \
                        feature_matching_loss(FFCN_feature, FFCN_feature_orig, train_batch[1])
                total_loss += feature_matching_loss_forward
                writer.add_scalar(
                    'loss/feature_matching_loss', feature_matching_loss_forward,
                    self.trainer.global_step)

            return total_loss

        # train discriminator
        if optimizer_idx == 1:
            Tensor = torch.FloatTensor  # if cuda else torch.FloatTensor
            valid = Variable(
                Tensor((out.shape[0])).fill_(1.0), requires_grad=False).unsqueeze(-1).to(self.device)
            fake = Variable(
                Tensor((out.shape[0])).fill_(0.0), requires_grad=False).unsqueeze(-1).to(self.device)

            if cfg['network_D']['netD'] == 'resnet3d':
                # 3d
                if cfg['train']['augmentation_method'] == "diffaug":
                    dis_real_loss = \
                        self.discriminator_criterion(
                            self.netD(DiffAugment(
                                torch.stack([train_batch[0], train_batch[2], train_batch[1]], dim=1),
                                cfg['train']['policy'])), valid)
                    dis_fake_loss = \
                        self.discriminator_criterion(
                            self.netD(torch.stack([train_batch[0], out, train_batch[1]], dim=1)),
                            fake)
                else:
                    dis_real_loss = \
                        self.discriminator_criterion(
                            self.netD(
                                torch.stack([train_batch[0], train_batch[2], train_batch[1]], dim=1)),
                            valid)
                    dis_fake_loss = \
                        self.discriminator_criterion(
                            self.netD(torch.stack([train_batch[0], out, train_batch[1]], dim=1)),
                            fake)
            else:
                # 2d
                if cfg['train']['augmentation_method'] == "diffaug":
                  if cfg["network_D"]['netD'] == "MultiscaleDiscriminator":
                    # fake
                    dis_fake_loss = 0
                    d_out = self.netD(DiffAugment(out, cfg['train']['policy']))
                    for i in d_out:
                      dis_fake_loss += self.discriminator_criterion(torch.mean(torch.mean(i[0], dim=2), dim=2), fake)
                    # real
                    dis_real_loss = 0
                    d_out = self.netD(DiffAugment(train_batch[2], cfg['train']['policy']))
                    for i in d_out:
                      dis_real_loss += self.discriminator_criterion(torch.mean(torch.mean(i[0], dim=2), dim=2), fake)
                  else:
                    discr_out_fake = self.netD(DiffAugment(out, cfg['train']['policy']))
                    discr_out_real = self.netD(DiffAugment(train_batch[2], cfg['train']['policy']))
                elif cfg['train']['augmentation_method'] == "MuarAugment":
                    self.mu_transform.setup(self)
                    if cfg["network_D"]['netD'] == "MultiscaleDiscriminator":
                      # fake
                      dis_fake_loss = 0
                      mu_augment, _ = self.mu_transform((out, fake))
                      d_out = self.netD(mu_augment)
                      for i in d_out:
                        dis_fake_loss += self.discriminator_criterion(torch.mean(torch.mean(i[0], dim=2), dim=2), fake)
                      # real
                      dis_real_loss = 0
                      mu_augment, _ = self.mu_transform((train_batch[2], valid))
                      d_out = self.netD(mu_augment)
                      for i in d_out:
                        dis_real_loss += self.discriminator_criterion(torch.mean(torch.mean(i[0], dim=2), dim=2), fake)
                    else:
                      # fake
                      mu_augment, _ = self.mu_transform((out, fake))
                      discr_out_fake = self.netD(mu_augment)
                      # real
                      mu_augment, _ = self.mu_transform((train_batch[2], valid))
                      discr_out_real = self.netD(mu_augment)
                elif cfg['train']['augmentation_method'] == "batch_aug":
                    fake_out, real_out = self.batch_aug(out, train_batch[2])
                    if cfg["network_D"]['netD'] == "MultiscaleDiscriminator":
                      # fake
                      dis_fake_loss = 0
                      d_out = self.netD(fake_out)
                      for i in d_out:
                        dis_fake_loss += self.discriminator_criterion(torch.mean(torch.mean(i[0], dim=2), dim=2), fake)
                      # real
                      dis_real_loss = 0
                      d_out = self.netD(real_out)
                      for i in d_out:
                        dis_real_loss += self.discriminator_criterion(torch.mean(torch.mean(i[0], dim=2), dim=2), fake)
                    else:
                      discr_out_fake = self.netD(fake_out)
                      discr_out_real = self.netD(real_out)
                else:
                    if cfg['network_D']['netD'] == 'FFCNLayerDiscriminator':
                        discr_out_fake, _ = self.netD(out)
                        discr_out_real, _ = self.netD(train_batch[2])
                    else:
                      if cfg["network_D"]['netD'] == "MultiscaleDiscriminator":
                        # fake
                        dis_fake_loss = 0
                        d_out = self.netD(out)
                        for i in d_out:
                          dis_fake_loss += self.discriminator_criterion(torch.mean(torch.mean(i[0], dim=2), dim=2), fake)
                        # real
                        dis_real_loss = 0
                        d_out = self.netD(out)
                        for i in d_out:
                          dis_real_loss += self.discriminator_criterion(torch.mean(torch.mean(i[0], dim=2), dim=2), fake)
                      else:
                        discr_out_fake = self.netD(out)
                        discr_out_real = self.netD(train_batch[2])

            # loss for multi does get calculated with a loop
            if cfg["network_D"]['netD'] != "MultiscaleDiscriminator":
              dis_fake_loss = self.discriminator_criterion(discr_out_fake.float(), fake.float())
              dis_real_loss = self.discriminator_criterion(discr_out_real.float(), fake.float())

            # Total loss
            d_loss = cfg["network_D"]["d_loss_weight"] * ((dis_real_loss + dis_fake_loss) / 2)

            writer.add_scalar('loss/d_loss', d_loss, self.trainer.global_step)

            return d_loss

    def configure_optimizers(self):
        if cfg['network_G']['finetune'] is True:
            input_G = self.netG.parameters()
        else:
            input_G = filter(lambda p: p.requires_grad, self.netG.parameters())

        if cfg['network_D']['netD'] is not None:
            input_D = self.netD.parameters()
        # perceptual loss does not get training and will be ignored

        if cfg['network_G']['finetune'] is None or cfg['network_G']['finetune'] is False:
            if cfg['train']['scheduler'] == 'Adam':
                opt_g = torch.optim.Adam(input_G, lr=cfg['train']['lr_g'])
                if cfg['network_D']['netD'] is not None:
                    opt_d = torch.optim.Adam(input_D, lr=cfg['train']['lr_d'])
            if cfg['train']['scheduler'] == 'AdamP':
                from adamp import AdamP
                opt_g = AdamP(
                    input_G,
                    lr=cfg['train']['lr_g'],
                    betas=(float(cfg['train']['betas0']), float(cfg['train']['betas1'])),
                    weight_decay=float(cfg['train']['weight_decay']))
                if cfg['network_D']['netD'] is not None:
                    opt_d = AdamP(
                        input_D, lr=cfg['train']['lr_d'],
                        betas=(float(cfg['train']['betas0']), float(cfg['train']['betas1'])),
                        weight_decay=float(cfg['train']['weight_decay']))
            if cfg['train']['scheduler'] == 'SGDP':
                from adamp import SGDP
                opt_g = SGDP(
                    input_G,
                    lr=cfg['train']['lr_g'],
                    weight_decay=cfg['train']['weight_decay'],
                    momentum=cfg['train']['momentum'],
                    nesterov=cfg['train']['nesterov'])
                if cfg['network_D']['netD'] is not None:
                    opt_d = SGDP(
                        input_D,
                        lr=cfg['train']['lr_d'],
                        weight_decay=cfg['train']['weight_decay'],
                        momentum=cfg['train']['momentum'],
                        nesterov=cfg['train']['nesterov'])
            if cfg['train']['scheduler'] == 'MADGRAD':
                from madgrad import MADGRAD
                opt_g = MADGRAD(
                    input_G,
                    lr=cfg['train']['lr_g'],
                    momentum=cfg['train']['momentum'],
                    weight_decay=cfg['train']['weight_decay'],
                    eps=cfg['train']['eps'])
                if cfg['network_D']['netD'] is not None:
                    opt_d = MADGRAD(
                        input_D,
                        lr=cfg['train']['lr_d'],
                        momentum=cfg['train']['momentum'],
                        weight_decay=cfg['train']['weight_decay'],
                        eps=cfg['train']['eps'])
            if cfg['train']['scheduler'] == 'cosangulargrad':
                from arch.optimizer.cosangulargrad import cosangulargrad
                opt_g = cosangulargrad(
                    input_G,
                    lr=cfg['train']['lr_g'],
                    betas=(float(cfg['train']['betas0']), float(cfg['train']['betas1'])),
                    eps=cfg['train']['eps'],
                    weight_decay=cfg['train']['weight_decay'])
                if cfg['network_D']['netD'] is not None:
                    opt_d = cosangulargrad(
                        input_D,
                        lr=cfg['train']['lr_d'],
                        betas=(float(cfg['train']['betas0']), float(cfg['train']['betas1'])),
                        eps=cfg['train']['eps'],
                        weight_decay=cfg['train']['weight_decay'])
            if cfg['train']['scheduler'] == 'tanangulargrad':
                from arch.optimizer.tanangulargrad import tanangulargrad
                opt_g = tanangulargrad(
                    input_G,
                    lr=cfg['train']['lr_g'],
                    betas=(float(cfg['train']['betas0']), float(cfg['train']['betas1'])),
                    eps=cfg['train']['eps'],
                    weight_decay=cfg['train']['weight_decay'])
                if cfg['network_D']['netD'] is not None:
                    opt_d = tanangulargrad(
                        input_D,
                        lr=cfg['train']['lr_d'],
                        betas=(float(cfg['train']['betas0']), float(cfg['train']['betas1'])),
                        eps=cfg['train']['eps'],
                        weight_decay=cfg['train']['weight_decay'])
            if cfg['train']['scheduler'] == 'Adam8bit':
                import bitsandbytes as bnb
                opt_g = bnb.optim.Adam8bit(
                    input_G,
                    lr=cfg['train']['lr_g'],
                    betas=(float(cfg['train']['betas0']), float(cfg['train']['betas1'])))
                if cfg['network_D']['netD'] is not None:
                    opt_d = bnb.optim.Adam8bit(
                        input_D,
                        lr=cfg['train']['lr_d'],
                        betas=(float(cfg['train']['betas0']), float(cfg['train']['betas1'])))
            if cfg['train']['scheduler'] == 'SGD_AGC':
                from nfnets import SGD_AGC
                opt_g = SGD_AGC(
                    input_G,
                    lr=cfg['train']['lr_g'],
                    weight_decay=cfg['train']['weight_decay'],
                    eps=cfg['train']['eps'])
                if cfg['network_D']['netD'] is not None:
                    opt_d = SGD_AGC(
                        input_D,
                        lr=cfg['train']['lr_d'],
                        weight_decay=cfg['train']['weight_decay'],
                        eps=cfg['train']['eps'])

        if cfg['train']['AGC'] is True:
            from nfnets.agc import AGC
            opt_g = AGC(input_G, opt_g)

        if cfg['network_D']['netD'] is not None:
            return [opt_g, opt_d], []
        else:
            return [opt_g], []

    def validation_step(self, train_batch, train_idx):
        # inpainting
        # train_batch[0] = masked (lr)
        # train_batch[1] = mask
        # train_batch[2] = path
        # train_batch[3] = edges
        # train_batch[4] = grayscale

        # super resolution
        # train_batch[0] = lr
        # train_batch[1] = hr
        # train_batch[2] = lr_path
        # train_batch[3] = landmarks (DFDNet)

        # frame interpolation
        # train_batch[0] = [img1, img3]
        # train_batch[1] = img2
        # train_batch[2] = imgpath

        if cfg['network_G']['netG'] == 'CTSDG':
            out, _, _ = self.netG(train_batch[0], train_batch[3], train_batch[1])
            out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

        # if frame interpolation
        if cfg['network_G']['netG'] in \
                ("CDFI", "sepconv_enhanced", 'CAIN', 'RRIN', 'ABME', 'EDSC'):
            out = self.netG(train_batch[0][0], train_batch[0][1])

        if cfg['network_G']['netG'] == 'rife':
            out, _ = self.netG(train_batch[0][0], train_batch[0][1], training=True)

        #########################

        if cfg['network_G']['netG'] in \
                ('lama', 'MST', 'MANet', 'context_encoder', 'aotgan', 'DFNet', 'AdaFill',
                 'MEDFE', 'RFR', 'LBAM', 'DMFN', 'Partial', 'RN', 'RN', 'DSNet',
                 'DSNetRRDB', 'DSNetDeoldify'):
            # generate fake (one output generator)
            out = self(train_batch[0], train_batch[1])
            # masking, taking original content from HR
            out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

        #########################

        if cfg['network_G']['netG'] in ('deepfillv1', 'deepfillv2', 'Adaptive'):
            # generate fake (two output generator)
            out, _ = self(train_batch[0], train_batch[1])
            # masking, taking original content from HR
            out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

        #########################

        # CSA
        if cfg['network_G']['netG'] == 'CSA':
            _, out, _, _ = self(train_batch[0], train_batch[1])
            # masking, taking original content from HR
            out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

        # EdgeConnect
        # train_batch[3] = edges
        # train_batch[4] = grayscale
        if cfg['network_G']['netG'] == 'EdgeConnect':
            out, _ = self.netG(train_batch[0], train_batch[3], train_batch[4], train_batch[1])

        # PVRS
        if cfg['network_G']['netG'] == 'PVRS':
            out, _, _, _ = self.netG(train_batch[0], train_batch[1], train_batch[3])

        # FRRN
        if cfg['network_G']['netG'] == 'FRRN':
            out, _, _ = self(train_batch[0], train_batch[1])

        # deoldify
        if cfg['network_G']['netG'] == 'deoldify':
            out = self.netG(train_batch[0])

        ############################
        # ESRGAN / GLEAN / GPEN / comodgan / lightweight_gan / ESRT / SRVGGNetCompact
        if cfg['network_G']['netG'] in \
                ("restormer", "SRVGGNetCompact", "ESRT", "swinir", 'lightweight_gan', 
                 'RRDB_net', 'GLEAN', 'GPEN', 'comodgan'):
            if (cfg['datasets']['train']['mode'] == 'DS_inpaint' 
                    or cfg['datasets']['train']['mode'] == 'DS_inpaint_tiled' 
                    or cfg['datasets']['train']['mode'] == 'DS_inpaint_tiled_batch'):
                # masked test with inpaint dataloader
                tmp = torch.cat([train_batch[0], train_batch[1]], 1)
                out = self.netG(tmp)
                out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])
            else:
                # normal dataloader
                out = self.netG(train_batch[0])

        # GFPGAN
        if cfg['network_G']['netG'] == 'GFPGAN':
            if (cfg['datasets']['train']['mode'] == 'DS_inpaint' 
                    or cfg['datasets']['train']['mode'] == 'DS_inpaint_tiled' 
                    or cfg['datasets']['train']['mode'] == 'DS_inpaint_tiled_batch'):
                # masked test with inpaint dataloader
                tmp = torch.cat([train_batch[0], train_batch[1]], 1)
                out, _ = self.netG(tmp)
                out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])
            else:
                out, _ = self.netG(train_batch[1])

        if cfg['network_G']['netG'] == 'srflow':
            from arch.SRFlowNet_arch import get_z
            # freeze rrdb in the beginning
            if self.trainer.global_step < cfg['network_G']['freeze_iter']:
                self.netG.set_rrdb_training(False)
            else:
                self.netG.set_rrdb_training(True)

            z = get_z(self, heat=0, seed=None, batch_size=train_batch[0].shape[0], lr_shape=train_batch[0].shape)
            out, logdet = self.netG(lr=train_batch[0], z=z, eps_std=0, reverse=True, reverse_with_grad=True)

        # DFDNet
        if cfg['network_G']['netG'] == 'DFDNet':
            out = self.netG(train_batch[0], part_locations=train_batch[3])
            # range [-1, 1] to [0, 1]
            out = out + 1
            out = out - out.min()
            out = out / (out.max() - out.min())

        # Validation metrics work, but they need an origial source image.
        if 'PSNR' in cfg['train']['metrics']:
            self.val_psnr.append(self.psnr_metric(train_batch[1], out).item())
        if 'SSIM' in cfg['train']['metrics']:
            self.val_ssim.append(self.ssim_metric(train_batch[1], out).item())
        if 'MSE' in cfg['train']['metrics']:
            self.val_mse.append(self.mse_metric(train_batch[1], out).item())
        if 'LPIPS' in cfg['train']['metrics']:
            self.val_lpips.append(self.PerceptualLoss(out, train_batch[1]).item())

        validation_output = cfg['path']['validation_output_path']

        # train_batch[2] can contain multiple files, depending on the batch_size
        for f in train_batch[2]:
            # data is processed as a batch, to save indididual files, a counter is used
            counter = 0
            if not os.path.exists(os.path.join(validation_output, os.path.splitext(os.path.basename(f))[0])):
                os.makedirs(os.path.join(validation_output, os.path.splitext(os.path.basename(f))[0]))

            filename_with_extention = os.path.basename(f)
            filename = os.path.splitext(filename_with_extention)[0]

            # if its yuv (cain), currently only supports batch_size 1
            if cfg['network_G']['netG'] in ("sepconv_enhanced", 'CAIN', 'rife'):
                out = out.data.mul(255).mul(255 / 255).clamp(0, 255).round()
                out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()  # *255
                out = out.astype(np.uint8)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(validation_output, filename,
                                 str(self.trainer.global_step) + '.png'), out)
            else:
                save_image(
                    out[counter], os.path.join(validation_output, filename,
                                               str(self.trainer.global_step) + '.png'))

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
