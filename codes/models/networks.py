import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from options.options import opt_get

#import models.modules.sft_arch as sft_arch
logger = logging.getLogger('base')
####################
# initialize networks
####################

def weights_init_normal(m, bias_fill=0, mean=0.0, std=0.02):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1 and classname != "DiscConvBlock": #ASRResNet's DiscConvBlock causes confusion
    if isinstance(m, nn.Conv2d):
        # init.normal_(m.weight.data, 0.0, std)
        init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            m.bias.data.fill_(bias_fill)
    # elif classname.find('Linear') != -1:
    elif isinstance(m, nn.Linear):
        # init.normal_(m.weight.data, 0.0, std)
        init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            m.bias.data.fill_(bias_fill)
    # elif classname.find('BatchNorm2d') != -1:
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        init.normal_(m.weight.data, mean=1.0, std=std)  # BN also uses norm
        if m.bias is not None:
            # init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(bias_fill)


def weights_init_kaiming(m, scale=1, bias_fill=0, **kwargs):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1 and classname != "DiscConvBlock": #ASRResNet's DiscConvBlock causes confusion
    if isinstance(m, nn.Conv2d):
        # init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight, **kwargs)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.fill_(bias_fill)
    # elif classname.find('Linear') != -1:
    elif isinstance(m, nn.Linear):
        # init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight, **kwargs)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.fill_(bias_fill)
    # elif classname.find('BatchNorm2d') != -1:
    # elif isinstance(m, _BatchNorm):
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        # init.constant_(m.weight.data, 1.0)
        init.constant_(m.weight, 1)
        if m.bias is not None:
            # init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(bias_fill)


def weights_init_orthogonal(m, bias_fill=0, **kwargs):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    if isinstance(m, nn.Conv2d):
        # init.orthogonal_(m.weight.data, gain=1)
        init.orthogonal_(m.weight.data, **kwargs)
        if m.bias is not None:
            m.bias.data.fill_(bias_fill)
    # elif classname.find('Linear') != -1:
    elif isinstance(m, nn.Linear):
        # init.orthogonal_(m.weight.data, gain=1)
        init.orthogonal_(m.weight.data, **kwargs)
        if m.bias is not None:
            m.bias.data.fill_(bias_fill)
    # elif classname.find('BatchNorm2d') != -1:
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        # init.constant_(m.weight.data, 1.0)
        init.constant_(m.weight, 1)
        if m.bias is not None:
            # init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(bias_fill)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt, step=0):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if opt_net['net_act']: # If set, use a different activation function
        act_type = opt_net['net_act']
    else: # Use networks defaults
        if which_model == 'sr_resnet':
            act_type = 'relu'
        elif which_model == 'RRDB_net':
            act_type = 'leakyrelu'
        elif which_model == 'ppon':
            act_type = 'leakyrelu'

    if which_model == 'sr_resnet':  # SRResNet
        from models.modules.architectures import SRResNet_arch
        netG = SRResNet_arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type=act_type, mode=opt_net['mode'], upsample_mode='pixelshuffle', \
            convtype=opt_net['convtype'], finalact=opt_net['finalact'])
    elif which_model == 'sft_arch':  # SFT-GAN
        from models.modules.architectures import sft_arch
        netG = sft_arch.SFT_Net()
    elif which_model == 'RRDB_net':  # RRDB
        from models.modules.architectures import RRDBNet_arch
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type=act_type, mode=opt_net['mode'], upsample_mode='upconv', convtype=opt_net['convtype'], \
            finalact=opt_net['finalact'], gaussian_noise=opt_net['gaussian'], plus=opt_net['plus'])
    elif which_model == 'MRRDB_net':  # Modified RRDB
        from models.modules.architectures import RRDBNet_arch
        netG = RRDBNet_arch.MRRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], gc=opt_net['gc'])
    elif which_model == 'ppon':
        from models.modules.architectures import PPON_arch
        netG = PPON_arch.PPON(in_nc=opt_net['in_nc'], nf=opt_net['nf'], nb=opt_net['nb'], out_nc=opt_net['out_nc'],
            upscale=opt_net['scale'], act_type=act_type) #(in_nc=3, nf=64, nb=24, out_nc=3)
    elif which_model == 'asr_cnn':
        from models.modules.architectures import ASRResNet_arch
        netG = ASRResNet_arch.ASRCNN(upscale_factor=opt_net['scale'], spectral_norm = True, self_attention = True, max_pool=True, poolsize = 4, finalact='tanh')
    elif which_model == 'asr_resnet':
        from models.modules.architectures import ASRResNet_arch
        netG = ASRResNet_arch.ASRResNet(scale_factor=opt_net['scale'], spectral_norm = True, self_attention = True, max_pool=True, poolsize = 4)
    elif which_model == 'abpn_net':
        from models.modules.architectures import ABPN_arch
        netG = ABPN_arch.ABPN_v5(input_dim=3, dim=32)
        # netG = ABPN_arch.ABPN_v5(input_dim=opt_net['in_nc'], dim=opt_net['out_nc'])
    elif which_model == 'pan_net': #PAN
        from models.modules.architectures import PAN_arch
        netG = PAN_arch.PAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                            nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'],
                            self_attention=opt_net.get('self_attention', False),
                            double_scpa=opt_net.get('double_scpa', False),
                            ups_inter_mode=opt_net.get('ups_inter_mode', 'nearest'))
    elif which_model == 'sofvsr_net':
        from models.modules.architectures import SOFVSR_arch
        netG = SOFVSR_arch.SOFVSR(scale=opt_net['scale'],n_frames=opt_net.get('n_frames', 3),
                                  channels=opt_net.get('channels', 320), img_ch=opt_net.get('img_ch', 1),
                                  SR_net=opt_net.get('SR_net', 'sofvsr'),
                                  sr_nf=opt_net.get('sr_nf', 64), sr_nb=opt_net.get('sr_nb', 23),
                                  sr_gc=opt_net.get('sr_gc', 32), sr_unf=opt_net.get('sr_unf', 24),
                                  sr_gaussian_noise=opt_net.get('sr_gaussian_noise', 64),
                                  sr_plus=opt_net.get('sr_plus', False), sr_sa=opt_net.get('sr_sa', True),
                                  sr_upinter_mode=opt_net.get('sr_upinter_mode', 'nearest'))
    elif which_model == 'rife_net':
        from models.modules.architectures import RIFE_arch
        netG = RIFE_arch.RIFE()
    elif which_model == 'DFNet':
        from models.modules.architectures import DFNet_arch
        netG = DFNet_arch.DFNet(c_img=opt_net['c_img'], c_mask=opt_net['c_mask'], c_alpha=opt_net['c_alpha'],
            mode=opt_net['mode'], norm=opt_net['norm'], act_en=opt_net['act_en'], act_de=opt_net['act_de'],
            en_ksize=opt_net['en_ksize'], de_ksize=opt_net['de_ksize'],
            blend_layers=opt_net['blend_layers'])
    elif which_model == 'EdgeConnect':
        from models.modules.architectures import EdgeConnect_arch
        netG = EdgeConnect_arch.EdgeConnectModel(use_spectral_norm=opt_net['use_spectral_norm'])
    elif which_model == 'CSA':
        from models.modules.architectures import CSA_arch
        netG = CSA_arch.InpaintNet()
    elif which_model == 'RN':
        from models.modules.architectures import RN_arch
        netG = RN_arch.G_Net(input_channels=opt_net['input_channels'], residual_blocks=opt_net['residual_blocks'], threshold=opt_net['threshold'])
        # using rn init to avoid errors
        RN_arch = RN_arch.rn_initialize_weights(netG, scale=0.1)
    elif which_model == 'deepfillv1':
        from models.modules.architectures import deepfillv1_arch
        netG = deepfillv1_arch.InpaintSANet()
    elif which_model == 'deepfillv2':
        from models.modules.architectures import deepfillv2_arch
        netG = deepfillv2_arch.GatedGenerator(in_channels = opt_net['in_channels'], out_channels = opt_net['out_channels'], latent_channels = opt_net['latent_channels'], pad_type = opt_net['pad_type'], activation = opt_net['activation'], norm = opt_net['norm'])
        # using deepfill init to avoid errors
        deepfillv2_arch.deepfillv2_weights_init(netG)
    elif which_model == 'Adaptive':
        from models.modules.architectures import Adaptive_arch
        netG = Adaptive_arch.PyramidNet(in_channels=opt_net['in_channels'], residual_blocks=opt_net['residual_blocks'], init_weights=opt_net['init_weights'])
    elif which_model == 'Global':
        from models.modules.architectures import Global_arch
        netG = Global_arch.Generator(input_dim=opt_net['input_dim'], ngf=opt_net['ngf'], use_cuda=opt_net['use_cuda'], device_ids=opt_net['device_ids'])
    elif which_model == 'Pluralistic':
        from models.modules.architectures import Pluralistic_arch
        netG = Pluralistic_arch.PluralisticGenerator(ngf_E=opt_net['ngf_E'], z_nc_E=opt_net['z_nc_E'], img_f_E=opt_net['img_f_E'], layers_E=opt_net['layers_E'], norm_E=opt_net['norm_E'], activation_E=opt_net['activation_E'],
                ngf_G=opt_net['ngf_G'], z_nc_G=opt_net['z_nc_G'], img_f_G=opt_net['img_f_G'], L_G=opt_net['L_G'], output_scale_G=opt_net['output_scale_G'], norm_G=opt_net['norm_G'], activation_G=opt_net['activation_G'])
        # using pluralistic init to avoid errors
        Pluralistic_arch.pluralistic_init_weights(netG, init_type='kaiming', gain=0.02)
    elif which_model == 'SRFlowNet':
        from models.modules.architectures import SRFlowNet_arch
        netG = SRFlowNet_arch.SRFlowNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt['scale'], K=opt_net['flow']['K'], opt=opt, step=step)
    elif which_model == 'sisr':
        from models.modules.architectures import sisr_arch
        netG = sisr_arch.EdgeSRModel(use_spectral_norm=opt_net['use_spectral_norm'])
    elif which_model == 'crfill':
        from models.modules.architectures import crfill_arch
        netG = crfill_arch.InpaintGenerator(cnum=opt_net['cnum'])
    elif which_model == 'DeepDFNet':
        from models.modules.architectures import DeepDFNet_arch
        netG = DeepDFNet_arch.GatedGenerator(in_channels = opt_net['in_channels'], out_channels = opt_net['out_channels'], latent_channels = opt_net['latent_channels'], pad_type = opt_net['pad_type'], activation = opt_net['activation'], norm = opt_net['norm'])
        # using deepfill init to avoid errors
        DeepDFNet_arch.deepfillv2_weights_init(netG)
    elif which_model == 'partial':
        from models.modules.architectures import partial_arch
        netG = partial_arch.PartialConv()
    elif which_model == 'DMFN':
        from models.modules.architectures import DMFN_arch
        netG = DMFN_arch.InpaintingGenerator(in_nc=opt_net['in_nc'],
          out_nc=opt_net['out_nc'],nf=opt_net['nf'],n_res=opt_net['n_res'],
          norm=opt_net['norm'], activation=opt_net['activation'])
    elif which_model == 'pennet':
        from models.modules.architectures import pennet_arch
        netG = pennet_arch.InpaintGenerator()
    elif which_model == 'LBAM':
        from models.modules.architectures import LBAM_arch
        netG = LBAM_arch.LBAMModel()
    elif which_model == 'RFR':
        from models.modules.architectures import RFR_arch
        netG = RFR_arch.RFRNet()
    elif which_model == 'FRRN':
        from models.modules.architectures import FRRN_arch
        netG = FRRN_arch.FRRNet()
    elif which_model == 'PRVS':
        from models.modules.architectures import PRVS_arch
        netG = PRVS_arch.PRVSNet()
    elif which_model == 'CRA':
        from models.modules.architectures import CRA_arch
        netG = CRA_arch.GatedGenerator()
    elif which_model == 'USRNet':
        from models.modules.architectures import USRNet_arch
        netG = USRNet_arch.USRNet()
    elif which_model == 'atrous':
        from models.modules.architectures import atrous_arch
        netG = atrous_arch.AtrousInpainter()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if opt['is_train'] and which_model != 'MRRDB_net' and which_model != 'RN' and which_model != 'Pluralistic'and which_model != 'deepfillv2' and which_model != 'DeepDFNet':
        # Note: MRRDB_net initializes the modules during init, no need to initialize again here
        # pluralistic, rn and deepfillv2 already does init in a different place
        init_weights(netG, init_type='kaiming', scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG


# Discriminator
def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']
    which_model_G = opt_net['which_model_G']

    if which_model_G == 'ppon':
        model_G = 'PPON'
    else:
        model_G = 'ESRGAN'

    if which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        from models.modules.architectures import sft_arch
        netD = sft_arch.ACD_VGG_BN_96()
    elif which_model == 'discriminator_vgg_96':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
    elif which_model == 'discriminator_vgg_128_SN':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_128_SN()
    elif which_model == 'discriminator_vgg_128':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
    elif which_model == 'discriminator_vgg_192' or which_model == 'discriminator_192': #vic in PPON its called Discriminator_192, instead of BasicSR's Discriminator_VGG_192
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
    elif which_model == 'discriminator_vgg_256' or which_model == 'discriminator_256':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_256(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
    elif which_model == 'discriminator_vgg': # General adaptative case
        from models.modules.architectures import discriminators
        try:
            size = int(opt['datasets']['train']['HR_size'])
            netD = discriminators.Discriminator_VGG(size=size, in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
        except ValueError:
            raise ValueError('VGG Discriminator size could not be parsed from the HR patch size. Check that the image patch size is either a power of 2 or 3 multiplied by a power of 2.')
    elif which_model == 'adiscriminator':
        from models.modules.architectures import ASRResNet_arch
        netD = ASRResNet_arch.ADiscriminator(spectral_norm=opt_net['spectral_norm'], self_attention=opt_net['self_attention'], \
            max_pool=opt_net['max_pool'], poolsize=opt_net['poolsize'])
    elif which_model == 'adiscriminator_s':
        from models.modules.architectures import ASRResNet_arch
        netD = ASRResNet_arch.ADiscriminator_S(spectral_norm=opt_net['spectral_norm'], self_attention=opt_net['self_attention'], \
            max_pool=opt_net['max_pool'], poolsize=opt_net['poolsize'] )
    elif which_model == 'discriminator_vgg_128_fea': #VGG-like discriminator with features extraction
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_128_fea(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], \
            convtype=opt_net['convtype'], arch=model_G, spectral_norm=opt_net['spectral_norm'], self_attention=opt_net['self_attention'], \
            max_pool=opt_net['max_pool'], poolsize=opt_net['poolsize'])
    elif which_model == 'discriminator_vgg_fea': #VGG-like discriminator with features extraction
        from models.modules.architectures import discriminators
        try:
            size = int(opt['datasets']['train']['HR_size'])
            netD = discriminators.Discriminator_VGG_fea(size=size, in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], \
                convtype=opt_net['convtype'], arch=model_G, spectral_norm=opt_net['spectral_norm'], self_attention=opt_net['self_attention'], \
                max_pool=opt_net['max_pool'], poolsize=opt_net['poolsize'])
        except ValueError:
            raise ValueError('VGG Discriminator size could not be parsed from the HR patch size. Check that the image patch size is either a power of 2 or 3 multiplied by a power of 2.')
    elif which_model == 'patchgan' or which_model == 'NLayerDiscriminator':
        from models.modules.architectures import discriminators
        netD = discriminators.NLayerDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'], n_layers=opt_net['nlayer'])
    elif which_model == 'pixelgan' or which_model == 'PixelDiscriminator':
        from models.modules.architectures import discriminators
        netD = discriminators.PixelDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'])
    elif which_model == 'multiscale':
        from models.modules.architectures import discriminators
        netD = discriminators.MultiscaleDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'], \
            n_layers=opt_net['nlayer'], num_D=opt_net['num_D'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    """
    elif which_model.startswith('discriminator_vgg_'): # User-defined case
        models.modules.architectures import discriminators
        vgg_size = which_model[18:]
        try:
            size = int(vgg_size)
            netD = discriminators.Discriminator_VGG(size=size, in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
        except ValueError:
            raise ValueError('VGG Discriminator size [{:s}] could not be parsed.'.format(vgg_size))
    #"""
    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD


def define_F(opt, use_bn=False):
    from models.modules.architectures import perceptual

    feat_network = 'vgg' #opt['feat_network'] #can be configurable option

    gpu_ids = opt['gpu_ids']
    if opt['datasets']['train']['znorm']:
        z_norm = opt['datasets']['train']['znorm']
    else:
        z_norm = False
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34

    if feat_network == 'resnet': #ResNet
        netF = perceptual.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    else: #VGG network (default)
        netF = perceptual.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
            use_input_norm=True, device=device, z_norm=z_norm)

    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF


####################
# model coversions and validation for
# network loading
####################

def normal2mod(state_dict):
    if 'model.0.weight' in state_dict:
        try:
            logger.info('Converting and loading an RRDB model to modified RRDB')
        except:
            print('Converting and loading an RRDB model to modified RRDB')
        crt_net = {}
        items = []

        for k, v in state_dict.items():
            items.append(k)

        # # directly copy
        # for k, v in crt_net.items():
        #     if k in state_dict and state_dict[k].size() == v.size():
        #         crt_net[k] = state_dict[k]
        #         items.remove(k)

        crt_net['conv_first.weight'] = state_dict['model.0.weight']
        crt_net['conv_first.bias'] = state_dict['model.0.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('model.1.sub.', 'RRDB_trunk.')
                if '.0.weight' in k:
                    ori_k = ori_k.replace('.0.weight', '.weight')
                elif '.0.bias' in k:
                    ori_k = ori_k.replace('.0.bias', '.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net['trunk_conv.weight'] = state_dict['model.1.sub.23.weight']
        crt_net['trunk_conv.bias'] = state_dict['model.1.sub.23.bias']
        crt_net['upconv1.weight'] = state_dict['model.3.weight']
        crt_net['upconv1.bias'] = state_dict['model.3.bias']
        crt_net['upconv2.weight'] = state_dict['model.6.weight']
        crt_net['upconv2.bias'] = state_dict['model.6.bias']
        crt_net['HRconv.weight'] = state_dict['model.8.weight']
        crt_net['HRconv.bias'] = state_dict['model.8.bias']
        crt_net['conv_last.weight'] = state_dict['model.10.weight']
        crt_net['conv_last.bias'] = state_dict['model.10.bias']
        state_dict = crt_net

    return state_dict

def mod2normal(state_dict):
    if 'conv_first.weight' in state_dict:
        try:
            logger.info('Converting and loading a modified RRDB model to normal RRDB')
        except:
            print('Converting and loading a modified RRDB model to normal RRDB')
        crt_net = {}
        items = []
        for k, v in state_dict.items():
            items.append(k)

        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net['model.1.sub.23.weight'] = state_dict['trunk_conv.weight']
        crt_net['model.1.sub.23.bias'] = state_dict['trunk_conv.bias']
        crt_net['model.3.weight'] = state_dict['upconv1.weight']
        crt_net['model.3.bias'] = state_dict['upconv1.bias']
        crt_net['model.6.weight'] = state_dict['upconv2.weight']
        crt_net['model.6.bias'] = state_dict['upconv2.bias']
        crt_net['model.8.weight'] = state_dict['HRconv.weight']
        crt_net['model.8.bias'] = state_dict['HRconv.bias']
        crt_net['model.10.weight'] = state_dict['conv_last.weight']
        crt_net['model.10.bias'] = state_dict['conv_last.bias']
        state_dict = crt_net
    return state_dict

def model_val(opt_net=None, state_dict=None, model_type=None):
    if model_type == 'G':
        model = opt_get(opt_net, ['network_G', 'which_model_G'])
        if model == 'RRDB_net': # tonormal
            return mod2normal(state_dict)
        elif model == 'MRRDB_net': # tomod
            return normal2mod(state_dict)
        else:
            return state_dict
    elif model_type == 'D':
        # no particular Discriminator validation at the moment
        # model = opt_get(opt_net, ['network_G', 'which_model_D'])
        return state_dict
    else:
        # if model_type not provided, return unchanged
        # (can do other validations here)
        return state_dict
