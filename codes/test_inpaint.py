import torch
import cv2
from torchvision.utils import save_image
import yaml
import argparse
import numpy as np

def preparing(args, netG, val_L, mask, device):
    netG.load_state_dict(torch.load(args.model_path))
    netG = netG.to(device)

    if args.fp16_mode == True:
      netG.half()
      val_L = val_L.type(torch.cuda.HalfTensor)
      mask = mask.type(torch.cuda.HalfTensor)
    return netG, val_L, mask

def edge(val_L):
    val_L_gray = cv2.cvtColor(val_L, cv2.COLOR_BGR2GRAY)
    val_L_canny = cv2.Canny(val_L_gray,100,150)

    val_L_canny = torch.from_numpy(val_L_canny).unsqueeze(0).permute(0,3,1,2)/255
    val_L_gray = torch.from_numpy(val_L_gray).unsqueeze(0).permute(0,3,1,2)/255
    return val_L_gray, val_L_canny

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input image.')
    parser.add_argument('--mask', type=str, help='Mask image (black/white). If a mask is not provided, then green from an input image will be extracted and used as a mask instead.')
    parser.add_argument('--output', type=str, required=True, help='Output path.')
    parser.add_argument('--yaml', type=str, required=True, help='config path (yaml).')
    parser.add_argument('--model_path', type=str, required=True, help='Model path.')
    parser.add_argument('--fp16_mode', type=bool, default=False, required=False)
    args = parser.parse_args()

    with open(args.yaml, "r") as ymlfile:
      cfg = yaml.safe_load(ymlfile)
    opt_net = cfg['network_G']
    which_model = opt_net['which_model_G']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_L = cv2.imread(args.input)
    val_L = cv2.cvtColor(val_L, cv2.COLOR_BGR2RGB)

    if args.mask is not None:
      mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    else:
      mask = 1-np.all(val_L == [0,255,0], axis=-1).astype(int)

    val_L = torch.from_numpy(val_L).unsqueeze(0).permute(0,3,1,2)/255
    mask = torch.from_numpy(mask).unsqueeze(2).unsqueeze(0).permute(0,3,1,2) #/255

    # deleting area that is marked with mask
    val_L = val_L*mask

    val_L = val_L.to(device)
    mask = mask.to(device)

    if which_model == 'AdaFill':
      from models.modules.architectures import AdaFill_arch
      netG = AdaFill_arch.InpaintNet()

    elif which_model == 'MEDFE':
      from models.modules.architectures import MEDFE_arch
      netG = MEDFE_arch.MEDFEGenerator()

    elif which_model == 'RFR':
      from models.modules.architectures import RFR_arch
      netG = RFR_arch.RFRNet(conv_type=opt_net['conv_type'])

    elif which_model == 'LBAM':
      from models.modules.architectures import LBAM_arch
      netG = LBAM_arch.LBAMModel(inputChannels=opt_net['inputChannels'], outputChannels=opt_net['outputChannels'])

    elif which_model == 'DMFN':
      from models.modules.architectures import DMFN_arch
      netG = DMFN_arch.InpaintingGenerator(in_nc=opt_net['in_nc'],
        out_nc=opt_net['out_nc'],nf=opt_net['nf'],n_res=opt_net['n_res'],
        norm=opt_net['norm'], activation=opt_net['activation'])

    elif which_model == 'partial':
      from models.modules.architectures import partial_arch
      netG = partial_arch.Model()

    elif which_model == 'Adaptive':
      from models.modules.architectures import Adaptive_arch
      netG = Adaptive_arch.PyramidNet(in_channels=opt_net['in_channels'], residual_blocks=opt_net['residual_blocks'], init_weights=opt_net['init_weights'])

    elif which_model == 'DFNet':
      from models.modules.architectures import DFNet_arch
      netG = DFNet_arch.DFNet(c_img=opt_net['c_img'], c_mask=opt_net['c_mask'], c_alpha=opt_net['c_alpha'],
          mode=opt_net['mode'], norm=opt_net['norm'], act_en=opt_net['act_en'], act_de=opt_net['act_de'],
          en_ksize=opt_net['en_ksize'], de_ksize=opt_net['de_ksize'],
          blend_layers=opt_net['blend_layers'], conv_type=opt_net['conv_type'])

    elif which_model == 'RN':
      from models.modules.architectures import RN_arch
      netG = RN_arch.G_Net(inPConvUNetput_channels=opt_net['input_channels'], residual_blocks=opt_net['residual_blocks'], threshold=opt_net['threshold'])




    # 2 rgb images
    elif which_model == 'CRA':
      from models.modules.architectures import CRA_arch
      netG = CRA_arch.GatedGenerator(activation=opt_net['activation'], norm=opt_net['norm'])

    elif which_model == 'pennet':
      from models.modules.architectures import pennet_arch
      netG = pennet_arch.InpaintGenerator()

    elif which_model == 'deepfillv1':
      from models.modules.architectures import deepfillv1_arch
      netG = deepfillv1_arch.InpaintSANet()

    elif which_model == 'deepfillv2':
      from models.modules.architectures import deepfillv2_arch
      netG = deepfillv2_arch.GatedGenerator(in_channels = opt_net['in_channels'], out_channels = opt_net['out_channels'], latent_channels = opt_net['latent_channels'], pad_type = opt_net['pad_type'], activation = opt_net['activation'], norm = opt_net['norm'], conv_type = opt_net['conv_type'])

    elif which_model == 'Global':
      from models.modules.architectures import Global_arch
      netG = Global_arch.Generator(input_dim=opt_net['input_dim'], ngf=opt_net['ngf'], use_cuda=opt_net['use_cuda'], device_ids=opt_net['device_ids'])

    elif which_model == 'crfill':
      from models.modules.architectures import crfill_arch
      netG = crfill_arch.InpaintGenerator(cnum=opt_net['cnum'])

    elif which_model == 'DeepDFNet':
      from models.modules.architectures import DeepDFNet_arch
      netG = DeepDFNet_arch.GatedGenerator(in_channels = opt_net['in_channels'], out_channels = opt_net['out_channels'], latent_channels = opt_net['latent_channels'], pad_type = opt_net['pad_type'], activation = opt_net['activation'], norm = opt_net['norm'])




    # special
    elif which_model == 'Pluralistic':
      # todo
      from models.modules.architectures import Pluralistic_arch
      netG = Pluralistic_arch.PluralisticGenerator(ngf_E=opt_net['ngf_E'], z_nc_E=opt_net['z_nc_E'], img_f_E=opt_net['img_f_E'], layers_E=opt_net['layers_E'], norm_E=opt_net['norm_E'], activation_E=opt_net['activation_E'],
              ngf_G=opt_net['ngf_G'], z_nc_G=opt_net['z_nc_G'], img_f_G=opt_net['img_f_G'], L_G=opt_net['L_G'], output_scale_G=opt_net['output_scale_G'], norm_G=opt_net['norm_G'], activation_G=opt_net['activation_G'])

      netG, val_L, mask = preparing(args, netG, val_L, mask, device)
      netG = netG.to(device)

      fake, _, _ = netG(val_L, img_inverted, mask)

    elif which_model == 'EdgeConnect':
      from models.modules.architectures import EdgeConnect_arch
      netG = EdgeConnect_arch.EdgeConnectModel(residual_blocks_edge=opt_net['residual_blocks_edge'],
          residual_blocks_inpaint=opt_net['residual_blocks_inpaint'], use_spectral_norm=opt_net['use_spectral_norm'],
          conv_type_edge=opt_net['conv_type_edge'], conv_type_inpaint=opt_net['conv_type_inpaint'])

      val_L_gray, val_L_canny = edge(val_L)
      netG, val_L, mask = preparing(args, netG, val_L, mask, device)

      fake, _ = netG(val_L, val_L_canny, val_L_gray, mask)

    elif which_model == 'FRRN':
      from models.modules.architectures import FRRN_arch
      netG = FRRN_arch.FRRNet()
      netG, val_L, mask = preparing(args, netG, val_L, mask, device)

      fake, _, _ = netG(val_L, mask)

    elif which_model == 'PRVS':
      from models.modules.architectures import PRVS_arch
      netG = PRVS_arch.PRVSNet()

      val_L_gray, val_L_canny = edge(val_L)
      netG, val_L, mask = preparing(args, netG, val_L, mask, device)

      fake, _ ,_, _ = netG(val_L, mask, val_L_canny)

    elif which_model == 'CSA':
      from models.modules.architectures import CSA_arch
      netG = CSA_arch.InpaintNet(c_img=opt_net['c_img'],
      norm=opt_net['norm'], act_en=opt_net['act_en'], act_de=opt_net['act_de'])

      netG.load_state_dict(torch.load(args.model_path))
      netG, val_L, mask = preparing(args, netG, val_L, mask, device)

      _, fake, _, _ = netG(val_L, mask)

    elif which_model == 'atrous':
      from models.modules.architectures import atrous_arch
      netG = atrous_arch.AtrousInpainter()
      netG, val_L, mask = preparing(args, netG, val_L, mask, device)

      fake = netG(val_L)

    else:
      print("Selected model is not implemented.")


    if which_model == 'AdaFill' or which_model == 'MEDFE' or which_model == 'RFR' or which_model == 'LBAM' or which_model == 'DMFN' or which_model == 'partial' or which_model == 'Adaptive' or which_model == 'DFNet' or which_model == 'RN':
      netG.load_state_dict(torch.load(args.model_path))
      netG, val_L, mask = preparing(args, netG, val_L, mask, device)
      fake = netG(val_L, mask)

    if which_model == 'CRA' or which_model == 'pennet' or which_model == 'deepfillv1' or which_model == 'deepfillv2' or which_model == 'Global' or which_model == 'crfill' or which_model == 'DeepDFNet':
      netG.load_state_dict(torch.load(args.model_path))
      netG, val_L, mask = preparing(args, netG, val_L, mask, device)
      fake, _ = netG(val_L, mask)

    fake = val_L * mask + fake * (1-mask)
    save_image(fake, args.output)



if __name__ == "__main__":
    main()
