def CreateGenerator(cfg):
    # generators with one output, no AMP means nan loss during training
    if cfg['network_G']['netG'] == 'RRDB_net':
        from arch.rrdb_arch import RRDBNet
        netG = RRDBNet(
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
        netG = DFNet(
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
        netG = InpaintNet()

    # MEDFE (batch_size: 1, no AMP)
    elif cfg['network_G']['netG'] == 'MEDFE':
        from arch.MEDFE_arch import MEDFEGenerator
        netG = MEDFEGenerator()

    # RFR
    # conv_type = partial or deform
    # Warning: One testrun with deform resulted in Nan errors after ~60k iterations. It is also very slow.
    # 'partial' is recommended, since this is what the official implementation does use.
    elif cfg['network_G']['netG'] == 'RFR':
        from arch.RFR_arch import RFRNet
        netG = RFRNet(conv_type=cfg['network_G']['conv_type'])

    # LBAM
    elif cfg['network_G']['netG'] == 'LBAM':
        from arch.LBAM_arch import LBAMModel
        netG = LBAMModel(inputChannels=cfg['network_G']['inputChannels'],
                                outputChannels=cfg['network_G']['outputChannels'])

    # DMFN
    elif cfg['network_G']['netG'] == 'DMFN':
        from arch.DMFN_arch import InpaintingGenerator
        netG = InpaintingGenerator(in_nc=4, out_nc=3, nf=64, n_res=8,
                                        norm='in', activation='relu')

    # partial
    elif cfg['network_G']['netG'] == 'Partial':
        from arch.partial_arch import Model
        netG = Model()

    # RN
    elif cfg['network_G']['netG'] == 'RN':
        from arch.RN_arch import G_Net, rn_initialize_weights
        netG = G_Net(
            input_channels=cfg['network_G']['input_channels'], 
            residual_blocks=cfg['network_G']['residual_blocks'], 
            threshold=cfg['network_G']['threshold'])
        # using rn init to avoid errors
        if self.global_step == 0:
            RN_arch = rn_initialize_weights(netG, scale=0.1)

    # DSNet
    elif cfg['network_G']['netG'] == 'DSNet':
        from arch.DSNet_arch import DSNet
        netG = DSNet(
            layer_size=cfg['network_G']['layer_sizenr'],
            input_channels=cfg['network_G']['input_channels'],
            upsampling_mode=cfg['network_G']['upsampling_mode'])

    # context_encoder
    elif cfg['network_G']['netG'] == 'context_encoder':
        from arch.context_encoder_arch import Net_G
        netG = Net_G()

    # MANet
    elif cfg['network_G']['netG'] == 'MANet':
        from arch.MANet_arch import PConvUNet
        netG = PConvUNet()

    # GPEN
    elif cfg['network_G']['netG'] == 'GPEN':
        from arch.GPEN_arch import FullGenerator
        netG = FullGenerator(
            input_channels=cfg['network_G']['input_channels'],
            style_dim=cfg['network_G']['style_dim'],
            n_mlp=cfg['network_G']['n_mlp'],
            channel_multiplier=cfg['network_G']['channel_multiplier'],
            blur_kernel=cfg['network_G']['blur_kernel'],
            lr_mlp=cfg['network_G']['lr_mlp'])

    # comodgan
    elif cfg['network_G']['netG'] == 'comodgan':
        from arch.comodgan_arch import Generator
        netG = Generator(
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
        netG = SwinIR(
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
        netG = DSNetRRDB(layer_size=8, input_channels=3, upsampling_mode='nearest',
                                in_nc=4, out_nc=3, nf=128, nb=8, gc=32, upscale=1,
                                norm_type=None, act_type='leakyrelu', mode='CNA',
                                upsample_mode='upconv', convtype='Conv2D', finalact=None,
                                gaussian_noise=True, plus=False, nr=3)

    # DSNetDeoldify
    elif cfg['network_G']['netG'] == 'DSNetDeoldify':
        from arch.experimental.DSNetDeoldify_arch import DSNetDeoldify
        netG = DSNetDeoldify()

    elif cfg['network_G']['netG'] == 'lightweight_gan':
        from arch.experimental.lightweight_gan_arch import Generator
        netG = Generator(
            image_size=cfg['network_G']['image_size'],
            latent_dim=cfg['network_G']['latent_dim'],
            fmap_max=cfg['network_G']['fmap_max'],
            fmap_inverse_coef=cfg['network_G']['fmap_inverse_coef'],
            transparent=cfg['network_G']['transparent'],
            greyscale=cfg['network_G']['greyscale'],
            freq_chan_attn=cfg['network_G']['freq_chan_attn'])

    elif cfg['network_G']['netG'] == 'SimpleFontGenerator512':
        from arch.experimental.lightweight_gan_arch import SimpleFontGenerator512
        netG = SimpleFontGenerator512(
            image_size=cfg['network_G']['image_size'],
            latent_dim=cfg['network_G']['latent_dim'],
            fmap_max=cfg['network_G']['fmap_max'],
            fmap_inverse_coef=cfg['network_G']['fmap_inverse_coef'],
            transparent=cfg['network_G']['transparent'],
            greyscale=cfg['network_G']['greyscale'],
            freq_chan_attn=cfg['network_G']['freq_chan_attn'])

    elif cfg['network_G']['netG'] == 'SimpleFontGenerator256':
        from arch.experimental.lightweight_gan_arch import SimpleFontGenerator256
        netG = SimpleFontGenerator256(
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
        netG = InpaintSANet()

    # deepfillv2
    # conv_type = partial or deform
    elif cfg['network_G']['netG'] == 'deepfillv2':
        from arch.deepfillv2_arch import GatedGenerator
        netG = GatedGenerator(
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
        netG = PyramidNet(
            in_channels=cfg['network_G']['in_channels'],
            residual_blocks=cfg['network_G']['residual_blocks'],
            init_weights=cfg['network_G']['init_weights'])

    ############################
    # exotic generators

    # Pluralistic
    elif cfg['network_G']['netG'] == 'Pluralistic':
        from arch.Pluralistic_arch import PluralisticGenerator
        netG = PluralisticGenerator(
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
        netG = EdgeConnectModel(
            residual_blocks_edge=cfg['network_G']['residual_blocks_edge'],
            residual_blocks_inpaint=cfg['network_G']['residual_blocks_inpaint'],
            use_spectral_norm=cfg['network_G']['use_spectral_norm'],
            conv_type_edge=cfg['network_G']['conv_type_edge'],
            conv_type_inpaint=cfg['network_G']['conv_type_inpaint'])

    # FRRN
    elif cfg['network_G']['netG'] == 'FRRN':
        from arch.FRRN_arch import FRRNet
        netG = FRRNet()

    # PRVS
    elif cfg['network_G']['netG'] == 'PRVS':
        from arch.PRVS_arch import PRVSNet
        netG = PRVSNet()

    # CSA
    elif cfg['network_G']['netG'] == 'CSA':
        from arch.CSA_arch import InpaintNet
        netG = InpaintNet(
            c_img=cfg['network_G']['c_img'],
            norm=cfg['network_G']['norm'],
            act_en=cfg['network_G']['act_en'],
            act_de=cfg['network_G']['network_G'])

    # deoldify
    elif cfg['network_G']['netG'] == 'deoldify':
        from arch.Deoldify_arch import Unet34
        netG = Unet34()

    # GLEAN (does init itself)
    elif cfg['network_G']['netG'] == 'GLEAN':
        from arch.GLEAN_arch import GLEANStyleGANv2
        if cfg['network_G']['pretrained'] is False:
            netG = GLEANStyleGANv2(
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
            netG = GLEANStyleGANv2(
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
        netG = SRFlowNet(
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
        netG = UNetDictFace(64)

    # GFPGAN (error with init?)
    elif cfg['network_G']['netG'] == 'GFPGAN':
        from arch.GFPGAN_arch import GFPGANv1
        netG = GFPGANv1(
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
        netG = CAIN(cfg['network_G']['depth'])

    elif cfg['network_G']['netG'] == 'rife':
        from arch.rife_arch import IFNet
        netG = IFNet()

    elif cfg['network_G']['netG'] == 'RRIN':
        from arch.RRIN_arch import Net
        netG = Net()

    elif cfg['network_G']['netG'] == 'ABME':
        from arch.ABME_arch import ABME
        netG = ABME()

    elif cfg['network_G']['netG'] == 'EDSC':
        from arch.EDSC_arch import Network
        netG = Network()

    elif cfg['network_G']['netG'] == 'CTSDG':
        from arch.CTSDG_arch import Generator
        netG = Generator()

    elif cfg['network_G']['netG'] == 'MST':
        from arch.MST_arch import InpaintGateGenerator
        netG = InpaintGateGenerator()

    elif cfg['network_G']['netG'] == 'lama':
        from arch.lama_arch import FFCResNetGenerator
        netG = FFCResNetGenerator(4, 3)

    elif cfg['network_G']['netG'] == "ESRT":
        from arch.ESRT_arch import ESRT
        netG = ESRT(
            hiddenDim=cfg['network_G']['hiddenDim'], 
            mlpDim=cfg['network_G']['mlpDim'], 
            scaleFactor=cfg['scale'])

    elif cfg['network_G']['netG'] == 'sepconv_enhanced':
        from arch.sepconv_enhanced_arch import Network
        netG = Network()

    elif cfg['network_G']['netG'] == 'sepconv_rt':
        from arch.sepconv_realtime_arch import InterpolationNet
        netG = InterpolationNet(real_time=cfg['network_G']['real_time'], 
            device=cfg['network_G']['device'], 
            in_channels=cfg['network_G']['in_channels'], 
            out_channels=cfg['network_G']['out_channels'])

    elif cfg['network_G']['netG'] == "CDFI":
        from arch.CDFI_arch import AdaCoFNet
        netG = AdaCoFNet()

    elif cfg['network_G']['netG'] == "SRVGGNetCompact":
        from arch.SRVGGNetCompact_arch import SRVGGNetCompact
        netG = SRVGGNetCompact(
            num_in_ch=cfg['network_G']['num_in_ch'], 
            num_out_ch=cfg['network_G']['num_out_ch'], 
            num_feat=cfg['network_G']['num_feat'], 
            num_conv=cfg['network_G']['num_conv'], 
            upscale=cfg['scale'], 
            act_type=cfg['network_G']['act_type'],
            conv_mode=cfg['network_G']['conv_mode'])

    elif cfg['network_G']['netG'] == "restormer":
        from arch.restormer_arch import Restormer
        netG = Restormer(
            inp_channels=cfg['network_G']['inp_channels'],
            out_channels=cfg['network_G']['out_channels'],
            dim=cfg['network_G']['dim'],
            num_blocks=cfg['network_G']['num_blocks'],
            num_refinement_blocks=cfg['network_G']['num_refinement_blocks'],
            heads=cfg['network_G']['heads'],
            ffn_expansion_factor=cfg['network_G']['ffn_expansion_factor'],
            bias=cfg['network_G']['bias'],
            LayerNorm_type=cfg['network_G']['LayerNorm_type'])

    elif cfg['network_G']['netG'] == "swinir2":
        from arch.swinir2_arch import SwinIR
        netG = SwinIR(upscale=cfg['scale'], 
            img_size=(cfg['network_G']['img_size'], cfg['network_G']['img_size']), 
            window_size=cfg['network_G']['window_size'], 
            img_range=cfg['network_G']['img_range'], 
            depths=cfg['network_G']['depths'], 
            embed_dim=cfg['network_G']['embed_dim'], 
            num_heads=cfg['network_G']['num_heads'], 
            mlp_ratio=cfg['network_G']['mlp_ratio'], 
            upsampler=cfg['network_G']['upsampler'], 
            use_deformable_block=cfg['network_G']['use_deformable_block'])

    elif cfg['network_G']['netG'] == "misf":
        from arch.misf_arch import MISF
        netG = MISF(residual_blocks=cfg['network_G']['residual_blocks'], 
            use_spectral_norm=cfg['network_G']['use_spectral_norm'])

    ############################

    if cfg['network_G']['CEM'] is True:
        from arch.CEM import CEMnet
        CEM_conf = CEMnet.Get_CEM_Conf(cfg['scale'])
        CEM_conf.sigmoid_range_limit = cfg['network_G']['sigmoid_range_limit']
        if CEM_conf.sigmoid_range_limit:
            CEM_conf.input_range = [-1, 1] if z_norm else [0, 1]
        kernel = None  # note: could pass a kernel here, but None will use default cubic kernel
        CEM_net = CEMnet.CEMnet(CEM_conf, upscale_kernel=kernel)
        CEM_net.WrapArchitecture(only_padders=True)
        netG = CEM_net.WrapArchitecture(
            netG, training_patch_size=cfg['datasets']['train']['HR_size'])

    return netG