import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan' or model == 'srragan' or model == 'srragan_hfen' or model == 'lpips':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    # elif model == 'srragan_n2n':
        # from .SRRaGAN_n2n_model import SRRaGANModel as M
    elif model == 'ppon':
        from .ppon_model import PPONModel as M
    elif model == 'asrragan':
        from .ASRRaGAN_model import ASRRaGANModel as M
    elif model == 'vsrgan':
        from .VSR_model import VSRModel as M
    elif model == 'pbr':
        from .PBR_model import PBRModel as M
    elif model == 'USRNet':
        from .USRNet_model import USRNetmodel as M
    elif (model == 'EdgeConnect' or model == 'DFNet' or model == 'CSA' or model == 'RN' or model == 'deepfillv1'
      or model == 'deepfillv2' or model == 'Adaptive' or model == 'Global'
      or model == 'Pluralistic' or 'sisr' or model == 'crfill' or model == 'DeepDFNet'
      or model == 'partial' or model == 'DMFN' or model == 'pennet' or model == 'LBAM'
      or model == 'RFR' or model == 'FRRN' or model == 'PRVS' or model == 'CRA'
      or model == 'atrous' or model == 'MEDFE' or model == 'AdaFill'):
        from .inpaint_model import inpaintModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
