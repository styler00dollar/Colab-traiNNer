import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

from loss.loss import CharbonnierLoss, GANLoss, GradientPenaltyLoss, HFENLoss, TVLoss, GradientLoss, ElasticLoss, RelativeL1, L1CosineSim, ClipL1, MaskedL1Loss, MultiscalePixelLoss, FFTloss, OFLoss, L1_regularization, ColorLoss, AverageLoss, GPLoss, CPLoss, SPL_ComputeWithTrace, SPLoss, Contextual_Loss, StyleLoss
from loss.perceptual_loss import PerceptualLoss
from loss.metrics import *
from torchvision.utils import save_image
from torch.autograd import Variable
import pytorch_lightning as pl

from tensorboardX import SummaryWriter
writer = SummaryWriter(logdir=cfg['path']['log_path'])

from init import weights_init

import os


class CustomTrainClass(pl.LightningModule):
  def __init__(self):
    super().__init__()
    ############################
    # generators with one output, no AMP means nan loss during training
    if cfg['network_G']['netG'] == 'RRDB_net':
      from arch.rrdb_arch import RRDBNet
      self.netG = RRDBNet(in_nc=cfg['network_G']['in_nc'], out_nc=cfg['network_G']['out_nc'], nf=cfg['network_G']['nf'], nb=cfg['network_G']['nb'], gc=cfg['network_G']['gc'], upscale=cfg['scale'], norm_type=cfg['network_G']['norm_type'],
                  act_type=cfg['network_G']['net_act'], mode=cfg['network_G']['mode'], upsample_mode=cfg['network_G']['upsample_mode'], convtype=cfg['network_G']['convtype'],
                  finalact=cfg['network_G']['finalact'], gaussian_noise=cfg['network_G']['gaussian'], plus=cfg['network_G']['plus'],
                  nr=cfg['network_G']['nr'])

    # DFNet
    if cfg['network_G']['netG'] == 'DFNet':
      from arch.DFNet_arch import DFNet
      self.netG = DFNet(c_img=cfg['network_G']['c_img'], c_mask=cfg['network_G']['c_mask'], c_alpha=cfg['network_G']['c_alpha'],
              mode=cfg['network_G']['mode'], norm=cfg['network_G']['norm'], act_en=cfg['network_G']['act_en'], act_de=cfg['network_G']['act_de'],
              en_ksize=cfg['network_G']['en_ksize'], de_ksize=cfg['network_G']['de_ksize'],
              blend_layers=cfg['network_G']['blend_layers'], conv_type=cfg['network_G']['conv_type'])

    # AdaFill
    if cfg['network_G']['netG'] == 'AdaFill':
      from arch.AdaFill_arch import InpaintNet
      self.netG = InpaintNet()

    # MEDFE (batch_size: 1, no AMP)
    if cfg['network_G']['netG'] == 'MEDFE':
      from arch.MEDFE_arch import MEDFEGenerator
      self.netG = MEDFEGenerator()

    # RFR
    # conv_type = partial or deform
    # Warning: One testrun with deform resulted in Nan errors after ~60k iterations. It is also very slow.
    # 'partial' is recommended, since this is what the official implementation does use.
    if cfg['network_G']['netG'] == 'RFR':
      from arch.RFR_arch import RFRNet
      self.netG = RFRNet(conv_type=cfg['network_G']['conv_type'])

    # LBAM
    if cfg['network_G']['netG'] == 'LBAM':
      from arch.LBAM_arch import LBAMModel
      self.netG = LBAMModel(inputChannels=cfg['network_G']['inputChannels'], outputChannels=cfg['network_G']['outputChannels'])

    # DMFN
    if cfg['network_G']['netG'] == 'DMFN':
      from arch.DMFN_arch import InpaintingGenerator
      self.netG = InpaintingGenerator(in_nc=4, out_nc=3,nf=64,n_res=8,
          norm='in', activation='relu')

    # partial
    if cfg['network_G']['netG'] == 'Partial':
      from arch.partial_arch import Model
      self.netG = Model()


    # RN
    if cfg['network_G']['netG'] == 'RN':
      from arch.RN_arch import G_Net, rn_initialize_weights
      self.netG = G_Net(input_channels=cfg['network_G']['input_channels'], residual_blocks=cfg['network_G']['residual_blocks'], threshold=cfg['network_G']['threshold'])
      # using rn init to avoid errors
      if self.global_step == 0:
        RN_arch = rn_initialize_weights(self.netG, scale=0.1)


    # DSNet
    if cfg['network_G']['netG'] == 'DSNet':
      from arch.DSNet_arch import DSNet
      self.netG = DSNet(layer_size=cfg['network_G']['layer_sizenr'], input_channels=cfg['network_G']['input_channels'], upsampling_mode=cfg['network_G']['upsampling_mode'])

    # Experimental

    #DSNetRRDB
    if cfg['network_G']['netG'] == 'DSNetRRDB':
      from arch.experimental.DSNetRRDB_arch import DSNetRRDB
      self.netG = DSNetRRDB(layer_size=8, input_channels=3, upsampling_mode='nearest',
                  in_nc=4, out_nc=3, nf=128, nb=8, gc=32, upscale=1, norm_type=None,
                  act_type='leakyrelu', mode='CNA', upsample_mode='upconv', convtype='Conv2D',
                  finalact=None, gaussian_noise=True, plus=False,
                  nr=3)

    # DSNetDeoldify
    if cfg['network_G']['netG'] == 'DSNetDeoldify':
      from arch.experimental.DSNetDeoldify_arch import DSNetDeoldify
      self.netG = DSNetDeoldify()

    ############################

    # generators with two outputs

    # deepfillv1
    if cfg['network_G']['netG'] == 'deepfillv1':
      from arch.deepfillv1_arch import InpaintSANet
      self.netG = InpaintSANet()

    # deepfillv2
    # conv_type = partial or deform
    if cfg['network_G']['netG'] == 'deepfillv2':
      from arch.deepfillv2_arch import GatedGenerator
      self.netG = GatedGenerator(in_channels=cfg['network_G']['in_channels'], out_channels=cfg['network_G']['out_channels'],
      latent_channels=cfg['network_G']['latent_channels'], pad_type=cfg['network_G']['pad_type'], activation=cfg['network_G']['activation'], norm=cfg['network_G']['norm'], conv_type = cfg['network_G']['conv_type'])

    # Adaptive
    # [Warning] Adaptive does not like PatchGAN, Multiscale and ResNet.
    if cfg['network_G']['netG'] == 'Adaptive':
      from arch.Adaptive_arch import PyramidNet
      self.netG = PyramidNet(in_channels=cfg['network_G']['in_channels'], residual_blocks=cfg['network_G']['residual_blocks'], init_weights=cfg['network_G']['init_weights'])

    ############################
    # exotic generators

    # Pluralistic
    if cfg['network_G']['netG'] == 'Pluralistic':
      from arch.Pluralistic_arch import PluralisticGenerator
      self.netG = PluralisticGenerator(ngf_E=cfg['network_G']['ngf_E'], z_nc_E=cfg['network_G']['z_nc_E'], img_f_E=cfg['network_G']['img_f_E'], layers_E=cfg['network_G']['layers_E'], norm_E=cfg['network_G']['norm_E'], activation_E=cfg['network_G']['activation_E'],
                ngf_G=cfg['network_G']['ngf_G'], z_nc_G=cfg['network_G']['z_nc_G'], img_f_G=cfg['network_G']['img_f_G'], L_G=cfg['network_G']['L_G'], output_scale_G=cfg['network_G']['output_scale_G'], norm_G=cfg['network_G']['norm_G'], activation_G=cfg['network_G']['activation_G'])

    # EdgeConnect
    if cfg['network_G']['netG'] == 'EdgeConnect':
      from arch.EdgeConnect_arch import EdgeConnectModel
      #conv_type_edge: 'normal' # normal | partial | deform (has no spectral_norm)
      self.netG = EdgeConnectModel(residual_blocks_edge=cfg['network_G']['residual_blocks_edge'],
              residual_blocks_inpaint=cfg['network_G']['residual_blocks_inpaint'], use_spectral_norm=cfg['network_G']['use_spectral_norm'],
              conv_type_edge=cfg['network_G']['conv_type_edge'], conv_type_inpaint=cfg['network_G']['conv_type_inpaint'])

    # FRRN
    if cfg['network_G']['netG'] == 'FRRN':
      from arch.FRRN_arch import FRRNet
      self.netG = FRRNet()

    # PRVS
    if cfg['network_G']['netG'] == 'PRVS':
      from arch.PRVS_arch import PRVSNet
      self.netG = PRVSNet()

    # CSA
    if cfg['network_G']['netG'] == 'CSA':
      from arch.CSA_arch import InpaintNet
      self.netG = InpaintNet(c_img=cfg['network_G']['c_img'], norm=cfg['network_G']['norm'], act_en=cfg['network_G']['act_en'],
                                 act_de=cfg['network_G']['network_G'])

    # deoldify
    if cfg['network_G']['netG'] == 'deoldify':
      from arch.Deoldify_arch import Unet34
      self.netG = Unet34()

    if self.global_step == 0:
      weights_init(self.netG, 'kaiming')
    ############################


    # discriminators
    # size refers to input shape of tensor
    if cfg['network_D']['netD'] == 'context_encoder':
      from arch.discriminators import context_encoder
      self.netD = context_encoder()


    # VGG
    if cfg['network_D']['netD'] == 'Discriminator_VGG':
      from arch.discriminators import Discriminator_VGG
      self.netD = Discriminator_VGG(size=cfg['network_D']['size'], in_nc=cfg['network_D']['in_nc'], base_nf=cfg['network_D']['base_nf'], norm_type=cfg['network_D']['norm_type'], act_type=cfg['network_D']['act_type'], mode=cfg['network_D']['mode'], convtype=cfg['network_D']['convtype'], arch=cfg['network_D']['arch'])



    if cfg['network_D']['netD'] == 'Discriminator_VGG_fea':
      from arch.discriminators import Discriminator_VGG_fea
      self.netD = Discriminator_VGG_fea(size=cfg['network_D']['size'], in_nc=cfg['network_D']['in_nc'], base_nf=cfg['network_D']['base_nf'], norm_type=cfg['network_D']['norm_type'], act_type=cfg['network_D']['act_type'], mode=cfg['network_D']['mode'], convtype=cfg['network_D']['convtype'],
        arch=cfg['network_D']['arch'], spectral_norm=cfg['network_D']['spectral_norm'], self_attention = cfg['network_D']['self_attention'], max_pool=cfg['network_D']['max_pool'], poolsize = cfg['network_D']['poolsize'])

    if cfg['network_D']['netD'] == 'Discriminator_VGG_128_SN':
      from arch.discriminators import Discriminator_VGG_128_SN
      self.netD = Discriminator_VGG_128_SN()

    if cfg['network_D']['netD'] == 'VGGFeatureExtractor':
      from arch.discriminators import VGGFeatureExtractor
      self.netD = VGGFeatureExtractor(feature_layer=cfg['feature_layer']['feature_layer'],use_bn=cfg['network_D']['use_bn'],use_input_norm=cfg['network_D']['use_input_norm'],device=torch.device(cfg['network_D']['device']),z_norm=cfg['network_D']['z_norm'])

    # PatchGAN
    if cfg['network_D']['netD'] == 'NLayerDiscriminator':
      from arch.discriminators import NLayerDiscriminator
      self.netD = NLayerDiscriminator(input_nc=cfg['network_D']['input_nc'], ndf=cfg['network_D']['ndf'], n_layers=cfg['network_D']['n_layers'], norm_layer=cfg['network_D']['norm_layer'],
          use_sigmoid=cfg['network_D']['use_sigmoid'], getIntermFeat=cfg['network_D']['getIntermFeat'], patch=cfg['network_D']['patch'], use_spectral_norm=cfg['network_D']['use_spectral_norm'])

    # Multiscale
    if cfg['network_D']['netD'] == 'MultiscaleDiscriminator':
      from arch.discriminators import MultiscaleDiscriminator
      self.netD = MultiscaleDiscriminator(input_nc=cfg['network_D']['input_nc'], ndf=cfg['network_D']['ndf'], n_layers=cfg['network_D']['n_layers'], norm_layer=cfg['network_D']['norm_layer'],
                 use_sigmoid=cfg['network_D']['use_sigmoid'], num_D=cfg['network_D']['num_D'], getIntermFeat=cfg['network_D']['getIntermFeat'])

    # ResNet
    if cfg['network_D']['netD'] == 'Discriminator_ResNet_128':
      from arch.discriminators import Discriminator_ResNet_128
      self.netD = Discriminator_ResNet_128(in_nc=cfg['network_D']['in_nc'], base_nf=cfg['network_D']['base_nf'], norm_type=cfg['network_D']['norm_type'], act_type=cfg['network_D']['act_type'], mode=cfg['network_D']['mode'])

    if cfg['network_D']['netD'] == 'ResNet101FeatureExtractor':
      from arch.discriminators import ResNet101FeatureExtractor
      self.netD = ResNet101FeatureExtractor(use_input_norm=cfg['network_D']['use_input_norm'], device=torch.device(cfg['network_D']['device']), z_norm=cfg['network_D']['z_norm'])

    # MINC
    if cfg['network_D']['netD'] == 'MINCNet':
      from arch.discriminators import MINCNet
      self.netD = MINCNet()

    # Pixel
    if cfg['network_D']['netD'] == 'PixelDiscriminator':
      from arch.discriminators import PixelDiscriminator
      self.netD = PixelDiscriminator(input_nc=cfg['network_D']['input_nc'], ndf=cfg['network_D']['ndf'], norm_layer=cfg['network_D']['norm_layer'])

    # EfficientNet
    if cfg['network_D']['netD'] == 'EfficientNet':
      from efficientnet_pytorch import EfficientNet
      self.netD = EfficientNet.from_pretrained(cfg['network_D']['EfficientNet_pretrain'])

    # ResNeSt
    # ["resnest50", "resnest101", "resnest200", "resnest269"]
    if cfg['network_D']['netD'] == 'ResNeSt':
      if cfg['network_D']['ResNeSt_pretrain'] == 'resnest50':
        from arch.discriminators import resnest50
        self.netD = resnest50(pretrained=True)
      if cfg['network_D']['ResNeSt_pretrain'] == 'resnest101':
        from arch.discriminators import resnest101
        self.netD = resnest101(pretrained=True)

      if cfg['network_D']['ResNeSt_pretrain'] == 'resnest200':
        from arch.discriminators import resnest200
        self.netD = resnest200(pretrained=True)

      if cfg['network_D']['ResNeSt_pretrain'] == 'resnest269':
        from arch.discriminators import resnest269
        self.netD = resnest269(pretrained=True)

    # need fixing
    #FileNotFoundError: [Errno 2] No such file or directory: '../experiments/pretrained_models/VGG16minc_53.pth'
    #self.netD = MINCFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True, device=torch.device('cpu'))

    # Transformer (Warning: uses own init!)
    if cfg['network_D']['netD'] == 'TranformerDiscriminator':
      from arch.discriminators import TranformerDiscriminator
      self.netD  = TranformerDiscriminator(img_size=cfg['network_D']['img_size'], patch_size=cfg['network_D']['patch_size'], in_chans=cfg['network_D']['in_chans'], num_classes=cfg['network_D']['num_classes'], embed_dim=cfg['network_D']['embed_dim'], depth=cfg['network_D']['depth'],
                  num_heads=cfg['network_D']['num_heads'], mlp_ratio=cfg['network_D']['mlp_ratio'], qkv_bias=cfg['network_D']['qkv_bias'], qk_scale=cfg['network_D']['qk_scale'], drop_rate=cfg['network_D']['drop_rate'], attn_drop_rate=cfg['network_D']['attn_drop_rate'],
                  drop_path_rate=cfg['network_D']['drop_path_rate'], hybrid_backbone=cfg['network_D']['hybrid_backbone'], norm_layer=cfg['network_D']['norm_layer'])

    if cfg['network_D']['netD'] != 'TranformerDiscriminator':
      if self.global_step == 0:
        weights_init(self.netD, 'kaiming')


    # loss functions
    self.l1 = nn.L1Loss()

    self.HFENLoss = HFENLoss(loss_f=cfg['train']['loss_f'], kernel=cfg['train']['kernel'], kernel_size=cfg['train']['kernel_size'], sigma = cfg['train']['sigma'], norm = cfg['train']['norm'])
    self.ElasticLoss = ElasticLoss(a=cfg['train']['a'], reduction=cfg['train']['reduction_elastic'])
    self.RelativeL1 = RelativeL1(eps=cfg['train']['eps'], reduction=cfg['train']['reduction_realtive'])
    self.L1CosineSim = L1CosineSim(loss_lambda=cfg['train']['loss_lambda'], reduction=cfg['train']['reduction_L1CosineSim'])
    self.ClipL1 = ClipL1(clip_min=cfg['train']['clip_min'], clip_max=cfg['train']['clip_max'])

    if cfg['train']['loss_f'] == 'L1Loss':
      loss_f = torch.nn.L1Loss
    # todo

    self.FFTloss = FFTloss(loss_f = loss_f, reduction=cfg['train']['reduction_fft'])
    self.OFLoss = OFLoss()
    self.GPLoss = GPLoss(trace=cfg['train']['trace'], spl_denorm=cfg['train']['spl_denorm'])
    self.CPLoss = CPLoss(rgb=cfg['train']['rgb'], yuv=cfg['train']['yuv'], yuvgrad=cfg['train']['yuvgrad'], trace=cfg['train']['trace'], spl_denorm=cfg['train']['spl_denorm'], yuv_denorm=cfg['train']['yuv_denorm'])
    self.StyleLoss = StyleLoss()
    self.TVLoss = TVLoss(tv_type=cfg['train']['tv_type'], p = cfg['train']['p'])
    self.PerceptualLoss = PerceptualLoss(model=cfg['train']['model'], net=cfg['train']['net'], colorspace=cfg['train']['colorspace'], spatial=cfg['train']['spatial'], use_gpu=cfg['train']['use_gpu'], gpu_ids=cfg['train']['gpu_ids'])
    self.Contextual_Loss = Contextual_Loss(cfg['train']['layers_weights'], crop_quarter=cfg['train']['crop_quarter'], max_1d_size=cfg['train']['max_1d_size'],
        distance_type = cfg['train']['distance_type'], b=cfg['train']['b'], band_width=cfg['train']['band_width'],
        use_vgg = cfg['train']['use_vgg'], net = cfg['train']['net_contextual'], calc_type = cfg['train']['calc_type'])

    self.MSELoss = torch.nn.MSELoss()
    self.L1Loss = nn.L1Loss()

    # metrics
    self.psnr_metric = PSNR()
    self.ssim_metric = SSIM()
    self.ae_metric = AE()
    self.mse_metric = MSE()


  def forward(self, image, masks):
      return self.netG(image, masks)


  def training_step(self, train_batch, batch_idx):
      # inpainting:
      # train_batch[0][0] = batch_size
      # train_batch[0] = masked
      # train_batch[1] = mask (lr)
      # train_batch[2] = original

      # super resolution
      # train_batch[0] = Null
      # train_batch[1] = lr
      # train_batch[2] = hr

      if cfg['datasets']['train']['mode'] == 'DS_inpaint_tiled_batch':
        # reducing dimension
        train_batch[0] = torch.squeeze(train_batch[0], 0)
        train_batch[1] = torch.squeeze(train_batch[1], 0)
        train_batch[2] = torch.squeeze(train_batch[2], 0)

      # train generator
      ############################
      if cfg['network_G']['netG'] == 'DFNet' or cfg['network_G']['netG'] == 'AdaFill' or cfg['network_G']['netG'] == 'MEDFE' or cfg['network_G']['netG'] == 'RFR' or cfg['network_G']['netG'] == 'LBAM' or cfg['network_G']['netG'] == 'DMFN' or cfg['network_G']['netG'] == 'Partial' or cfg['network_G']['netG'] == 'RN' or cfg['network_G']['netG'] == 'RN' or cfg['network_G']['netG'] == 'DSNet' or cfg['network_G']['netG'] == 'DSNetRRDB' or cfg['network_G']['netG'] == 'DSNetDeoldify':
        # generate fake (1 output)
        out = self(train_batch[0],train_batch[1])

        # masking, taking original content from HR
        out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

      ############################
      if cfg['network_G']['netG'] == 'deepfillv1' or cfg['network_G']['netG'] == 'deepfillv2' or cfg['network_G']['netG'] == 'Adaptive':
        # generate fake (2 outputs)
        out, other_img = self(train_batch[0],train_batch[1])

        # masking, taking original content from HR
        out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

      ############################
      # exotic generators
      # CSA
      if cfg['network_G']['netG'] == 'CSA':
        coarse_result, out, csa, csa_d = self(train_batch[0],train_batch[1])
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
        out, _ ,edge_small, edge_big = self.netG(train_batch[0], train_batch[1], train_batch[3])
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
      # ESRGAN
      if cfg['network_G']['netG'] == 'RRDB_net':
        out = self.netG(train_batch[1])


      ############################
      # loss calculation
      total_loss = 0
      if cfg['train']['HFEN_weight'] > 0:
        HFENLoss_forward = cfg['train']['HFEN_weight']*self.HFENLoss(out, train_batch[2])
        total_loss += HFENLoss_forward
        writer.add_scalar('loss/HFEN', HFENLoss_forward, self.trainer.global_step)

      if cfg['train']['Elatic_weight'] > 0:
        ElasticLoss_forward = cfg['train']['Elatic_weight']*self.ElasticLoss(out, train_batch[2])
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

      if cfg['train']['Contexual_weight'] > 0:
        Contextual_Loss_forward = cfg['train']['Contexual_weight']*self.Contextual_Loss(out, train_batch[2])
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

      if cfg['train']['PerceptualLoss_weight'] > 0:
        perceptual_forward = cfg['train']['PerceptualLoss_weight']*self.PerceptualLoss(out, train_batch[2])
        total_loss += perceptual_forward
        writer.add_scalar('loss/perceptual', perceptual_forward, self.trainer.global_step)




      #########################
      # exotic loss

      # if model has two output, also calculate loss for such an image
      # example with just l1 loss
      if cfg['network_G']['netG'] == 'deepfillv1' or cfg['network_G']['netG'] == 'deepfillv2' or cfg['network_G']['netG'] == 'Adaptive':
        l1_stage1 = cfg['train']['stage1_weight']*self.L1Loss(other_img, train_batch[2])
        total_loss += l1_stage1
        writer.add_scalar('loss/l1_stage1', l1_stage1, self.trainer.global_step)


      # CSA Loss
      if cfg['network_G']['netG'] == 'CSA':
        recon_loss = self.L1Loss(coarse_result, train_batch[2]) + self.L1Loss(out, train_batch[2])
        cons = ConsistencyLoss()
        cons_loss = cons(csa, csa_d, train_batch[2], train_batch[1])
        self.log('loss/recon_loss', recon_loss)
        total_loss += recon_loss
        self.log('loss/cons_loss', cons_loss)
        total_loss += cons_loss
        writer.add_scalar('loss/recon_loss', recon_loss, self.trainer.global_step)
        writer.add_scalar('loss/cons_loss', cons_loss, self.trainer.global_step)

      # EdgeConnect
      # train_batch[3] = edges
      # train_batch[4] = grayscale
      if cfg['network_G']['netG'] == 'EdgeConnect':
        l1_edge = self.L1Loss(other_img, train_batch[3])
        self.log('loss/l1_edge', l1_edge)
        total_loss += l1_edge
        writer.add_scalar('loss/l1_edge', l1_edge, self.trainer.global_step)


      # PVRS
      if cfg['network_G']['netG'] == 'PVRS':
        edge_big_l1 = self.L1Loss(edge_big, train_batch[3])
        edge_small_l1 = self.L1Loss(edge_small, torch.nn.functional.interpolate(train_batch[3], scale_factor = 0.5))
        self.log('loss/edge_big_l1', edge_big_l1)
        total_loss += edge_big_l1
        self.log('loss/edge_small_l1', edge_small_l1)
        total_loss += edge_small_l1
        writer.add_scalar('loss/edge_big_l1', edge_big_l1, self.trainer.global_step)
        writer.add_scalar('loss/edge_small_l1', edge_small_l1, self.trainer.global_step)


      # FRRN
      if cfg['network_G']['netG'] == 'FRRN':
        mid_l1_loss = 0
        for idx in range(len(mid_x) - 1):
            mid_l1_loss += self.L1Loss(mid_x[idx] * mid_mask[idx], train_batch[2] * mid_mask[idx])
        self.log('loss/mid_l1_loss', mid_l1_loss)
        total_loss += mid_l1_loss
        writer.add_scalar('loss/mid_l1_loss', mid_l1_loss, self.trainer.global_step)


      #self.log('loss/g_loss', total_loss)
      writer.add_scalar('loss/g_loss', total_loss, self.trainer.global_step)

      #return total_loss
      #########################








      # train discriminator
      # resizing input if needed
      #train_batch[2] = torch.nn.functional.interpolate(train_batch[2], (128,128), align_corners=False, mode='bilinear')
      #out = torch.nn.functional.interpolate(out, (128,128), align_corners=False, mode='bilinear')

      Tensor = torch.cuda.FloatTensor #if cuda else torch.FloatTensor
      valid = Variable(Tensor(out.shape).fill_(1.0), requires_grad=False)
      fake = Variable(Tensor(out.shape).fill_(0.0), requires_grad=False)
      dis_real_loss = self.MSELoss(train_batch[2], valid)
      dis_fake_loss = self.MSELoss(out, fake)

      d_loss = (dis_real_loss + dis_fake_loss) / 2
      #self.log('loss/d_loss', d_loss)
      writer.add_scalar('loss/d_loss', d_loss, self.trainer.global_step)

      return total_loss+d_loss

  def configure_optimizers(self):
      if cfg['network_G']['finetune'] is None or cfg['network_G']['finetune'] == False:
        if cfg['train']['scheduler'] == 'Adam':
          optimizer = torch.optim.Adam(self.netG.parameters(), lr=cfg['train']['lr'])
        if cfg['train']['scheduler'] == 'AdamP':
          from adamp import AdamP
          optimizer = AdamP(self.netG.parameters(), lr=cfg['train']['lr'], betas=(float(cfg['train']['betas0']), float(cfg['train']['betas1'])), weight_decay=float(cfg['train']['weight_decay']))
        if cfg['train']['scheduler'] == 'SGDP':
          from adamp import SGDP
          optimizer = SGDP(self.netG.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'], momentum=cfg['train']['momentum'], nesterov=cfg['train']['nesterov'])

      if cfg['network_G']['finetune'] == True:
        if cfg['train']['scheduler'] == 'Adam':
          optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, self.netG.parameters()), lr=cfg['train']['lr'])
        if cfg['train']['scheduler'] == 'AdamP':
          from adamp import AdamP
          optimizer = AdamP(filter(lambda p:p.requires_grad, self.netG.parameters()), lr=cfg['train']['lr'], betas=(float(cfg['train']['betas0']), float(cfg['train']['betas1'])), weight_decay=float(cfg['train']['weight_decay']))
        if cfg['train']['scheduler'] == 'SGDP':
          from adamp import SGDP
          optimizer = SGDP(filter(lambda p:p.requires_grad, self.netG.parameters()), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'], momentum=cfg['train']['momentum'], nesterov=cfg['train']['nesterov'])

      return optimizer

  def validation_step(self, train_batch, train_idx):
    # inpainting
    # train_batch[0] = masked (lr)
    # train_batch[1] = mask
    # train_batch[2] = path

    # super resolution
    # train_batch[0] = lr
    # train_batch[1] = hr
    # train_batch[2] = lr_path

    #########################
    if cfg['network_G']['netG'] == 'aotgan' or cfg['network_G']['netG'] == 'DFNet' or cfg['network_G']['netG'] == 'AdaFill' or cfg['network_G']['netG'] == 'MEDFE' or cfg['network_G']['netG'] == 'RFR' or cfg['network_G']['netG'] == 'LBAM' or cfg['network_G']['netG'] == 'DMFN' or cfg['network_G']['netG'] == 'Partial' or cfg['network_G']['netG'] == 'RN' or cfg['network_G']['netG'] == 'RN' or cfg['network_G']['netG'] == 'DSNet' or cfg['network_G']['netG'] == 'DSNetRRDB' or cfg['network_G']['netG'] == 'DSNetDeoldify':
      # generate fake (one output generator)
      out = self(train_batch[0],train_batch[1])
      # masking, taking original content from HR
      out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])
    #########################
    if cfg['network_G']['netG'] == 'deepfillv1' or cfg['network_G']['netG'] == 'deepfillv2' or cfg['network_G']['netG'] == 'Adaptive':
      # generate fake (two output generator)
      out, _ = self(train_batch[0],train_batch[1])
      # masking, taking original content from HR
      out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])
    #########################
    # CSA
    if cfg['network_G']['netG'] == 'CSA':
      _, out, _, _ = self(train_batch[0],train_batch[1])
      # masking, taking original content from HR
      out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

    # EdgeConnect
    # train_batch[3] = edges
    # train_batch[4] = grayscale
    if cfg['network_G']['netG'] == 'EdgeConnect':
      out, _ = self.netG(train_batch[0], train_batch[3], train_batch[4], train_batch[1])

    # PVRS
    if cfg['network_G']['netG'] == 'PVRS':
      out, _ ,_, _ = self.netG(train_batch[0], train_batch[1], train_batch[3])

    # FRRN
    if cfg['network_G']['netG'] == 'FRRN':
      out, _, _ = self(train_batch[0], train_batch[1])

    # deoldify
    if cfg['network_G']['netG'] == 'deoldify':
      out = self.netG(train_batch[0])

    ############################
    # ESRGAN
    if cfg['network_G']['netG'] == 'RRDB_net':
      out = self.netG(train_batch[0])

    # Validation metrics work, but they need an origial source image.
    # Change dataloader to provide LR and HR if you want metrics.
    if 'PSNR' in cfg['train']['metrics']:
      self.log('metrics/PSNR', self.psnr_metric(train_batch[1], out))
    if 'SSIM' in cfg['train']['metrics']:
      self.log('metrics/SSIM', self.ssim_metric(train_batch[1], out))
    if 'MSE' in cfg['train']['metrics']:
      self.log('metrics/MSE', self.mse_metric(train_batch[1], out))
    if 'LPIPS' in cfg['train']['metrics']:
      self.log('metrics/LPIPS', self.PerceptualLoss(out, train_batch[1]))

    validation_output = cfg['path']['validation_output_path']

    # train_batch[2] can contain multiple files, depending on the batch_size
    for f in train_batch[2]:
      # data is processed as a batch, to save indididual files, a counter is used
      counter = 0
      if not os.path.exists(os.path.join(validation_output, os.path.splitext(os.path.basename(f))[0])):
        os.makedirs(os.path.join(validation_output, os.path.splitext(os.path.basename(f))[0]))

      filename_with_extention = os.path.basename(f)
      filename = os.path.splitext(filename_with_extention)[0]

      save_image(out[counter], os.path.join(validation_output, filename, str(self.trainer.global_step) + '.png'))

      counter += 1

  def test_step(self, train_batch, train_idx):
    # inpainting
    # train_batch[0] = masked
    # train_batch[1] = mask
    # train_batch[2] = path

    # super resolution
    # train_batch[0] = lr
    # train_batch[1] = hr
    # train_batch[2] = lr_path
    test_output = cfg['path']['test_output_path']
    if not os.path.exists(test_output):
      os.makedirs(test_output)

    out = self(train_batch[0].unsqueeze(0),train_batch[1].unsqueeze(0))
    out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

    save_image(out, os.path.join(test_output, os.path.splitext(os.path.basename(train_batch[2]))[0] + '.png'))
