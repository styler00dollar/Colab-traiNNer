from vic.loss import CharbonnierLoss, GANLoss, GradientPenaltyLoss, HFENLoss, TVLoss, GradientLoss, ElasticLoss, RelativeL1, L1CosineSim, ClipL1, MaskedL1Loss, MultiscalePixelLoss, FFTloss, OFLoss, L1_regularization, ColorLoss, AverageLoss, GPLoss, CPLoss, SPL_ComputeWithTrace, SPLoss, Contextual_Loss, StyleLoss
from vic.perceptual_loss import PerceptualLoss
from metrics import *
from torchvision.utils import save_image
from torch.autograd import Variable

from tensorboardX import SummaryWriter
logdir='/content/'
writer = SummaryWriter(logdir=logdir)

from adamp import AdamP
#from adamp import SGDP

class CustomTrainClass(pl.LightningModule):
  def __init__(self):
    super().__init__()
    ############################
    # generators with one output, no AMP means nan loss during training
    self.netG = RRDBNet(in_nc=3, out_nc=3, nf=128, nb=8, gc=32, upscale=4, norm_type=None,
                act_type='leakyrelu', mode='CNA', upsample_mode='upconv', convtype='Conv2D',
                finalact=None, gaussian_noise=True, plus=False,
                nr=3)

    # DFNet
    #self.netG = DFNet(c_img=3, c_mask=1, c_alpha=3,
    #        mode='nearest', norm='batch', act_en='relu', act_de='leaky_relu',
    #        en_ksize=[7, 5, 5, 3, 3, 3, 3, 3], de_ksize=[3, 3, 3, 3, 3, 3, 3, 3],
    #        blend_layers=[0, 1, 2, 3, 4, 5], conv_type='partial')

    # AdaFill
    #self.netG = InpaintNet()

    # MEDFE (batch_size: 1, no AMP)
    #self.netG = MEDFEGenerator()

    # RFR
    # conv_type = partial or deform
    # Warning: One testrun with deform resulted in Nan errors after ~60k iterations. It is also very slow.
    # 'partial' is recommended, since this is what the official implementation does use.
    #self.netG = RFRNet(conv_type='partial')

    # LBAM
    #self.netG = LBAMModel(inputChannels=4, outputChannels=3)

    # DMFN
    #self.netG = InpaintingGenerator(in_nc=4, out_nc=3,nf=64,n_res=8,
    #      norm='in', activation='relu')

    # partial
    #self.netG = Model()

    # RN
    #self.netG = G_Net(input_channels=3, residual_blocks=8, threshold=0.8)
    # using rn init to avoid errors
    #RN_arch = rn_initialize_weights(self.netG, scale=0.1)


    # DSNet
    #self.netG = DSNet(layer_size=8, input_channels=3, upsampling_mode='nearest')


    #DSNetRRDB
    #self.netG = DSNetRRDB(layer_size=8, input_channels=3, upsampling_mode='nearest',
    #            in_nc=4, out_nc=3, nf=128, nb=8, gc=32, upscale=1, norm_type=None,
    #            act_type='leakyrelu', mode='CNA', upsample_mode='upconv', convtype='Conv2D',
    #            finalact=None, gaussian_noise=True, plus=False,
    #            nr=3)


    # DSNetDeoldify
    #self.netG = DSNetDeoldify()

    ############################

    # generators with two outputs

    # deepfillv1
    #self.netG = InpaintSANet()

    # deepfillv2
    # conv_type = partial or deform
    #self.netG = GatedGenerator(in_channels=4, out_channels=3,
    #  latent_channels=64, pad_type='zero', activation='lrelu', norm='in', conv_type = 'partial')

    # Adaptive
    # [Warning] Adaptive does not like PatchGAN, Multiscale and ResNet.
    #self.netG = PyramidNet(in_channels=3, residual_blocks=1, init_weights='True')

    ############################
    # exotic generators

    # Pluralistic
    #self.netG = PluralisticGenerator(ngf_E=opt_net['ngf_E'], z_nc_E=opt_net['z_nc_E'], img_f_E=opt_net['img_f_E'], layers_E=opt_net['layers_E'], norm_E=opt_net['norm_E'], activation_E=opt_net['activation_E'],
    #            ngf_G=opt_net['ngf_G'], z_nc_G=opt_net['z_nc_G'], img_f_G=opt_net['img_f_G'], L_G=opt_net['L_G'], output_scale_G=opt_net['output_scale_G'], norm_G=opt_net['norm_G'], activation_G=opt_net['activation_G'])


    # EdgeConnect
    #conv_type_edge: 'normal' # normal | partial | deform (has no spectral_norm)
    #self.netG = EdgeConnectModel(residual_blocks_edge=8,
    #        residual_blocks_inpaint=8, use_spectral_norm=True,
    #        conv_type_edge='normal', conv_type_inpaint='normal')

    # FRRN
    #self.netG = FRRNet()

    # PRVS
    #self.netG = PRVSNet()

    # CSA
    #self.netG = InpaintNet(c_img=3, norm='instance', act_en='leaky_relu',
    #                           act_de='relu')

    # deoldify
    #self.netG = Unet34()

    weights_init(self.netG, 'kaiming')
    ############################


    # discriminators
    # size refers to input shape of tensor

    self.netD = context_encoder()

    # VGG
    #self.netD = Discriminator_VGG(size=256, in_nc=3, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN')
    #self.netD = Discriminator_VGG_fea(size=256, in_nc=3, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D',
    #     arch='ESRGAN', spectral_norm=False, self_attention = False, max_pool=False, poolsize = 4)
    #self.netD = Discriminator_VGG_128_SN()
    #self.netD = VGGFeatureExtractor(feature_layer=34,use_bn=False,use_input_norm=True,device=torch.device('cpu'),z_norm=False)

    # PatchGAN
    #self.netD = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
    #    use_sigmoid=False, getIntermFeat=False, patch=True, use_spectral_norm=False)

    # Multiscale
    #self.netD = MultiscaleDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
    #             use_sigmoid=False, num_D=3, getIntermFeat=False)

    # ResNet
    #self.netD = Discriminator_ResNet_128(in_nc=3, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA')
    #self.netD = ResNet101FeatureExtractor(use_input_norm=True, device=torch.device('cpu'), z_norm=False)

    # MINC
    #self.netD = MINCNet()

    # Pixel
    #self.netD = PixelDiscriminator(input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d)

    # EfficientNet
    #from efficientnet_pytorch import EfficientNet
    #self.netD = EfficientNet.from_pretrained('efficientnet-b0')

    # ResNeSt
    # ["resnest50", "resnest101", "resnest200", "resnest269"]
    #self.netD = resnest50(pretrained=True)

    # need fixing
    #FileNotFoundError: [Errno 2] No such file or directory: '../experiments/pretrained_models/VGG16minc_53.pth'
    #self.netD = MINCFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True, device=torch.device('cpu'))

    # Transformer (Warning: uses own init!)
    #self.netD  = TranformerDiscriminator(img_size=256, patch_size=1, in_chans=3, num_classes=1, embed_dim=64, depth=7,
    #             num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
    #             drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)


    weights_init(self.netD, 'kaiming')


    # loss functions
    self.l1 = nn.L1Loss()
    l_hfen_type = L1CosineSim()
    self.HFENLoss = HFENLoss(loss_f=l_hfen_type, kernel='log', kernel_size=15, sigma = 2.5, norm = False)
    self.ElasticLoss = ElasticLoss(a=0.2, reduction='mean')
    self.RelativeL1 = RelativeL1(eps=.01, reduction='mean')
    self.L1CosineSim = L1CosineSim(loss_lambda=5, reduction='mean')
    self.ClipL1 = ClipL1(clip_min=0.0, clip_max=10.0)
    self.FFTloss = FFTloss(loss_f = torch.nn.L1Loss, reduction='mean')
    self.OFLoss = OFLoss()
    self.GPLoss = GPLoss(trace=False, spl_denorm=False)
    self.CPLoss = CPLoss(rgb=True, yuv=True, yuvgrad=True, trace=False, spl_denorm=False, yuv_denorm=False)
    self.StyleLoss = StyleLoss()
    self.TVLoss = TVLoss(tv_type='tv', p = 1)
    self.PerceptualLoss = PerceptualLoss(model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0], model_path=None)
    layers_weights = {'conv_1_1': 1.0, 'conv_3_2': 1.0}
    self.Contextual_Loss = Contextual_Loss(layers_weights, crop_quarter=False, max_1d_size=100,
        distance_type = 'cosine', b=1.0, band_width=0.5,
        use_vgg = True, net = 'vgg19', calc_type = 'regular')

    self.MSELoss = torch.nn.MSELoss()
    self.L1Loss = nn.L1Loss()

    # metrics
    self.psnr_metric = PSNR()
    self.ssim_metric = SSIM()
    self.ae_metric = AE()
    self.mse_metric = MSE()


  def forward(self, image, masks):
      return self.netG(image, masks)

  #def adversarial_loss(self, y_hat, y):
  #    return F.binary_cross_entropy(y_hat, y)


  def training_step(self, train_batch, batch_idx):
      # inpainting:
      # train_batch[0][0] = batch_size
      # train_batch[0] = masked
      # train_batch[1] = mask
      # train_batch[2] = original

      # super resolution
      # train_batch[0] = lr
      # train_batch[1] = hr

      # train generator
      ############################
      # generate fake (1 output)
      #out = self(train_batch[0],train_batch[1])

      # masking, taking original content from HR
      #out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

      ############################
      # generate fake (2 outputs)
      #out, other_img = self(train_batch[0],train_batch[1])

      # masking, taking original content from HR
      #out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

      ############################
      # exotic generators
      # CSA
      #coarse_result, out, csa, csa_d = self(train_batch[0],train_batch[1])

      # EdgeConnect
      # train_batch[3] = edges
      # train_batch[4] = grayscale
      #out, other_img = self.netG(train_batch[0], train_batch[3], train_batch[4], train_batch[1])

      # PVRS
      #out, _ ,edge_small, edge_big = self.netG(train_batch[0], train_batch[1], train_batch[3])

      # FRRN
      #out, mid_x, mid_mask = self(train_batch[0], train_batch[1])

      # masking, taking original content from HR
      #out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

      # deoldify
      #out = self.netG(train_batch[0])


      ############################
      # ESRGAN
      out = self.netG(train_batch[0])



      ############################
      # loss calculation
      total_loss = 0
      """
      HFENLoss_forward = self.HFENLoss(out, train_batch[0])
      total_loss += HFENLoss_forward
      ElasticLoss_forward = self.ElasticLoss(out, train_batch[0])
      total_loss += ElasticLoss_forward
      RelativeL1_forward = self.RelativeL1(out, train_batch[0])
      total_loss += RelativeL1_forward
      """
      L1CosineSim_forward = 5*self.L1CosineSim(out, train_batch[1])
      total_loss += L1CosineSim_forward
      #self.log('loss/L1CosineSim', L1CosineSim_forward)
      writer.add_scalar('loss/L1CosineSim', L1CosineSim_forward, self.trainer.global_step)

      """
      ClipL1_forward = self.ClipL1(out, train_batch[0])
      total_loss += ClipL1_forward
      FFTloss_forward = self.FFTloss(out, train_batch[0])
      total_loss += FFTloss_forward
      OFLoss_forward = self.OFLoss(out)
      total_loss += OFLoss_forward
      GPLoss_forward = self.GPLoss(out, train_batch[0])
      total_loss += GPLoss_forward

      CPLoss_forward = 0.1*self.CPLoss(out, train_batch[0])
      total_loss += CPLoss_forward


      Contextual_Loss_forward = self.Contextual_Loss(out, train_batch[0])
      total_loss += Contextual_Loss_forward
      self.log('loss/contextual', Contextual_Loss_forward)
      """

      #style_forward = 240*self.StyleLoss(out, train_batch[2])
      #total_loss += style_forward
      #self.log('loss/style', style_forward)

      tv_forward = 0.0000005*self.TVLoss(out)
      total_loss += tv_forward
      #self.log('loss/tv', tv_forward)
      writer.add_scalar('loss/tv', tv_forward, self.trainer.global_step)

      perceptual_forward = 2*self.PerceptualLoss(out, train_batch[1])
      total_loss += perceptual_forward
      #self.log('loss/perceptual', perceptual_forward)
      writer.add_scalar('loss/perceptual', perceptual_forward, self.trainer.global_step)







      #########################
      # exotic loss

      # if model has two output, also calculate loss for such an image
      # example with just l1 loss

      #l1_stage1 = self.L1Loss(other_img, train_batch[0])
      #self.log('loss/l1_stage1', l1_stage1)
      #total_loss += l1_stage1


      # CSA Loss
      """
      recon_loss = self.L1Loss(coarse_result, train_batch[2]) + self.L1Loss(out, train_batch[2])
      cons = ConsistencyLoss()
      cons_loss = cons(csa, csa_d, train_batch[2], train_batch[1])
      self.log('loss/recon_loss', recon_loss)
      total_loss += recon_loss
      self.log('loss/cons_loss', cons_loss)
      total_loss += cons_loss
      """

      # EdgeConnect
      # train_batch[3] = edges
      # train_batch[4] = grayscale
      #l1_edge = self.L1Loss(other_img, train_batch[3])
      #self.log('loss/l1_edge', l1_edge)
      #total_loss += l1_edge

      # PVRS
      """
      edge_big_l1 = self.L1Loss(edge_big, train_batch[3])
      edge_small_l1 = self.L1Loss(edge_small, torch.nn.functional.interpolate(train_batch[3], scale_factor = 0.5))
      self.log('loss/edge_big_l1', edge_big_l1)
      total_loss += edge_big_l1
      self.log('loss/edge_small_l1', edge_small_l1)
      total_loss += edge_small_l1
      """

      # FRRN
      """
      mid_l1_loss = 0
      for idx in range(len(mid_x) - 1):
          mid_l1_loss += self.L1Loss(mid_x[idx] * mid_mask[idx], train_batch[2] * mid_mask[idx])
      self.log('loss/mid_l1_loss', mid_l1_loss)
      total_loss += mid_l1_loss
      """

      #self.log('loss/g_loss', total_loss)
      writer.add_scalar('loss/g_loss', total_loss, self.trainer.global_step)

      #return total_loss
      #########################








      # train discriminator
      # resizing input if needed
      #train_batch[2] = torch.nn.functional.interpolate(train_batch[2], (128,128), align_corners=False, mode='bilinear')
      #out = torch.nn.functional.interpolate(out, (128,128), align_corners=False, mode='bilinear')

      Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
      valid = Variable(Tensor(out.shape).fill_(1.0), requires_grad=False)
      fake = Variable(Tensor(out.shape).fill_(0.0), requires_grad=False)
      dis_real_loss = self.MSELoss(train_batch[1], valid)
      dis_fake_loss = self.MSELoss(out, fake)

      d_loss = (dis_real_loss + dis_fake_loss) / 2
      #self.log('loss/d_loss', d_loss)
      writer.add_scalar('loss/d_loss', d_loss, self.trainer.global_step)

      return total_loss+d_loss

  def configure_optimizers(self):
      #optimizer = torch.optim.Adam(self.netG.parameters(), lr=2e-3)
      optimizer = AdamP(self.netG.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-2)
      #optimizer = SGDP(self.netG.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)
      return optimizer

  def validation_step(self, train_batch, train_idx):
    # inpainting
    # train_batch[0] = masked
    # train_batch[1] = mask
    # train_batch[2] = path

    # super resolution
    # train_batch[0] = lr
    # train_batch[1] = hr
    # train_batch[2] = lr_path

    #########################
    # generate fake (one output generator)
    #out = self(train_batch[0],train_batch[1])
    # masking, taking original content from HR
    #out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

    #########################
    # generate fake (two output generator)
    #out, _ = self(train_batch[0],train_batch[1])

    # masking, taking original content from HR
    #out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])
    #########################
    # CSA
    #_, out, _, _ = self(train_batch[0],train_batch[1])
    # masking, taking original content from HR
    #out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

    # EdgeConnect
    # train_batch[3] = edges
    # train_batch[4] = grayscale
    #out, _ = self.netG(train_batch[0], train_batch[3], train_batch[4], train_batch[1])

    # PVRS
    #out, _ ,_, _ = self.netG(train_batch[0], train_batch[1], train_batch[3])

    # FRRN
    #out, _, _ = self(train_batch[0], train_batch[1])

    # deoldify
    #out = self.netG(train_batch[0])

    ############################
    # ESRGAN
    out = self.netG(train_batch[0])

    # Validation metrics work, but they need an origial source image.
    # Change dataloader to provide LR and HR if you want metrics.
    self.log('metrics/PSNR', self.psnr_metric(train_batch[1], out))
    self.log('metrics/SSIM', self.ssim_metric(train_batch[1], out))
    self.log('metrics/MSE', self.mse_metric(train_batch[1], out))
    self.log('metrics/LPIPS', self.PerceptualLoss(out, train_batch[1]))

    validation_output = '/content/validation_output/' #@param

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
    test_output = '/content/test_output/' #@param
    if not os.path.exists(test_output):
      os.makedirs(test_output)

    out = self(train_batch[0].unsqueeze(0),train_batch[1].unsqueeze(0))
    out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

    save_image(out, os.path.join(test_output, os.path.splitext(os.path.basename(train_batch[2]))[0] + '.png'))
