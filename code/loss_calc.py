import pytorch_lightning as pl
from loss.loss import (
    FocalFrequencyLoss,
    feature_matching_loss,
    FrobeniusNormLoss,
    LapLoss,
    HFENLoss,
    TVLoss,
    GradientLoss,
    ElasticLoss,
    RelativeL1,
    L1CosineSim,
    ClipL1,
    MultiscalePixelLoss,
    FFTloss,
    OFLoss,
    YUVColorLoss,
    XYZColorLoss,
    GPLoss,
    CPLoss,
    SPLoss,
    Contextual_Loss,
    StyleLoss,
    ConsistencyLoss,
    CannyLoss,
    KullbackHistogramLoss,
    KullbackHistogramLossV2,
    SalientRegionLoss,
    glcmLoss,
    GradientDomainLoss,
    SobelLoss,
    ColorHarmonyLoss,
    VIT_FeatureLoss,
    VIT_MMD_FeatureLoss,
    TIMM_FeatureLoss,
    LaplacianLoss,
    SobelLossV2,
    textured_loss,
    ldl_loss,
    IQA_loss,
)

import torch
import torch.nn as nn
import os
from torch.autograd import Variable


class AllLoss(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # loss functions
        if cfg["train"]["L1Loss_weight"] > 0:
            self.l1 = nn.L1Loss()

        if cfg["train"]["HFEN_weight"] > 0:
            if cfg["train"]["loss_f"] == "L1Loss":
                loss_f_hfen = torch.nn.L1Loss()
            elif cfg["train"]["loss_f"] == "L1CosineSim":
                loss_f_hfen = L1CosineSim(
                    loss_lambda=cfg["train"]["loss_lambda"],
                    reduction=cfg["train"]["reduction_L1CosineSim"],
                )
            self.HFENLoss = HFENLoss(
                loss_f=loss_f_hfen,
                kernel=cfg["train"]["kernel"],
                kernel_size=cfg["train"]["kernel_size"],
                sigma=cfg["train"]["sigma"],
                norm=cfg["train"]["norm"],
            )

        if cfg["train"]["Elastic_weight"] > 0:
            self.ElasticLoss = ElasticLoss(
                a=cfg["train"]["a"], reduction=cfg["train"]["reduction_elastic"]
            )

        if cfg["train"]["Relative_l1_weight"] > 0:
            self.RelativeL1 = RelativeL1(
                eps=cfg["train"]["l1_eps"], reduction=cfg["train"]["reduction_relative"]
            )

        if cfg["train"]["L1CosineSim_weight"] > 0:
            self.L1CosineSim = L1CosineSim(
                loss_lambda=cfg["train"]["loss_lambda"],
                reduction=cfg["train"]["reduction_L1CosineSim"],
            )

        if cfg["train"]["ClipL1_weight"] > 0:
            self.ClipL1 = ClipL1(
                clip_min=cfg["train"]["clip_min"], clip_max=cfg["train"]["clip_max"]
            )

        if cfg["train"]["FFTLoss_weight"] > 0:
            if cfg["train"]["loss_f_fft"] == "L1Loss":
                loss_f_fft = torch.nn.L1Loss
            elif cfg["train"]["loss_f_fft"] == "L1CosineSim":
                loss_f_fft = L1CosineSim(
                    loss_lambda=cfg["train"]["loss_lambda"],
                    reduction=cfg["train"]["reduction_L1CosineSim"],
                )
            self.FFTloss = FFTloss(
                loss_f=loss_f_fft, reduction=cfg["train"]["reduction_fft"]
            )

        if cfg["train"]["OFLoss_weight"] > 0:
            self.OFLoss = OFLoss()

        if cfg["train"]["GPLoss_weight"] > 0:
            self.GPLoss = GPLoss(
                trace=cfg["train"]["gp_trace"], spl_denorm=cfg["train"]["gp_spl_denorm"]
            )
        if cfg["train"]["CPLoss_weight"] > 0:
            self.CPLoss = CPLoss(
                rgb=cfg["train"]["rgb"],
                yuv=cfg["train"]["yuv"],
                yuvgrad=cfg["train"]["yuvgrad"],
                trace=cfg["train"]["cp_trace"],
                spl_denorm=cfg["train"]["cp_spl_denorm"],
                yuv_denorm=cfg["train"]["yuv_denorm"],
            )

        if cfg["train"]["StyleLoss_weight"] > 0:
            self.StyleLoss = StyleLoss()

        if cfg["train"]["TVLoss_weight"] > 0:
            self.TVLoss = TVLoss(tv_type=cfg["train"]["tv_type"], p=cfg["train"]["p"])

        if cfg["train"]["Contextual_weight"] > 0:
            self.Contextual_Loss = Contextual_Loss(
                cfg["train"]["layers_weights"],
                crop_quarter=cfg["train"]["crop_quarter"],
                max_1d_size=cfg["train"]["max_1d_size"],
                distance_type=cfg["train"]["distance_type"],
                b=cfg["train"]["b"],
                band_width=cfg["train"]["band_width"],
                use_vgg=cfg["train"]["use_vgg"],
                net=cfg["train"]["net_contextual"],
                calc_type=cfg["train"]["calc_type"],
                use_timm=cfg["train"]["use_timm"],
                timm_model=cfg["train"]["timm_model"],
            )

        if cfg["train"]["textured_loss_weight"] > 0:
            self.textured_loss = textured_loss()

        if cfg["train"]["MSE_weight"] > 0:
            self.MSELoss = torch.nn.MSELoss()

        if cfg["train"]["L1Loss_weight"] > 0:
            self.L1Loss = nn.L1Loss()

        if cfg["train"]["BCE_weight"] > 0:
            self.BCELogits = torch.nn.BCEWithLogitsLoss()
            self.BCE = torch.nn.BCELoss()

        if cfg["train"]["FFLoss_weight"] > 0:
            self.FFLoss = FocalFrequencyLoss()

        # perceptual loss
        if cfg["train"]["perceptual_weight"] > 0:
            from arch.networks_basic import PNetLin

            self.perceptual_loss = PNetLin(
                pnet_rand=cfg["train"]["pnet_rand"],
                pnet_tune=cfg["train"]["pnet_tune"],
                pnet_type=cfg["train"]["pnet_type"],
                use_dropout=cfg["train"]["use_dropout"],
                spatial=cfg["train"]["spatial"],
                version=cfg["train"]["version"],
                lpips=cfg["train"]["lpips"],
            )
            model_path = os.path.abspath(
                f'loss/lpips_weights/v0.1/{cfg["train"]["pnet_type"]}.pth'
            )
            print(f"Loading model from: {model_path}")
            self.perceptual_loss.load_state_dict(
                torch.load(model_path, map_location=torch.device(self.device)),
                strict=False,
            )
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False

            if (
                cfg["train"]["force_fp16_perceptual"] is True
                and cfg["train"]["perceptual_tensorrt"] is False
            ):
                print("Converting perceptual model to FP16")
                self.perceptual_loss = self.perceptual_loss.half()

            if (
                cfg["train"]["force_fp16_perceptual"] is True
                and cfg["train"]["perceptual_tensorrt"] is True
            ):
                print("Converting perceptual model to TensorRT (FP16)")
                import torch_tensorrt

                example_data = torch.rand(1, 3, 256, 448).half().cuda()
                self.perceptual_loss = self.perceptual_loss.half().cuda()
                self.perceptual_loss = torch.jit.trace(
                    self.perceptual_loss, [example_data, example_data]
                )
                self.perceptual_loss = torch_tensorrt.compile(
                    self.perceptual_loss,
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=(1, 3, 64, 64),
                            opt_shape=(1, 3, 256, 448),
                            max_shape=(1, 3, 720, 1280),
                            dtype=torch.half,
                        ),
                        torch_tensorrt.Input(
                            min_shape=(1, 3, 64, 64),
                            opt_shape=(1, 3, 256, 448),
                            max_shape=(1, 3, 720, 1280),
                            dtype=torch.half,
                        ),
                    ],
                    enabled_precisions={torch.half},
                    truncate_long_and_double=True,
                )
                del example_data

            elif (
                cfg["train"]["force_fp16_perceptual"] is False
                and cfg["train"]["perceptual_tensorrt"] is True
            ):
                print("Converting perceptual model to TensorRT")
                import torch_tensorrt

                example_data = torch.rand(1, 3, 256, 448)
                self.perceptual_loss = torch.jit.trace(
                    self.perceptual_loss, [example_data, example_data]
                )
                self.perceptual_loss = torch_tensorrt.compile(
                    self.perceptual_loss,
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=(1, 3, 64, 64),
                            opt_shape=(1, 3, 256, 448),
                            max_shape=(1, 3, 720, 1280),
                            dtype=torch.float32,
                        ),
                        torch_tensorrt.Input(
                            min_shape=(1, 3, 64, 64),
                            opt_shape=(1, 3, 256, 448),
                            max_shape=(1, 3, 720, 1280),
                            dtype=torch.float32,
                        ),
                    ],
                    enabled_precisions={torch.float},
                    truncate_long_and_double=True,
                )
                del example_data

        if cfg["network_G"]["netG"] == "CSA":
            self.ConsistencyLoss = ConsistencyLoss()

        if cfg["train"]["Canny_weight"] > 0:
            self.CannyLoss = CannyLoss(
                threshold=cfg["train"]["canny_threshold"],
                blurred_img_weight=cfg["train"]["canny_blurred_img_weight"],
                grad_mag_weight=cfg["train"]["canny_grad_mag_weight"],
                grad_orientation_weight=cfg["train"]["canny_grad_mag_weight"],
                thin_edges_weight=cfg["train"]["canny_thin_edges_weight"],
                thresholded_weight=cfg["train"]["canny_thresholded_weight"],
                early_threshold=cfg["train"]["canny_early_threshold"],
            )

        if cfg["train"]["KullbackHistogramLoss_weight"] > 0:
            self.KullbackHistogramLoss = KullbackHistogramLoss()

        if cfg["train"]["KullbackHistogramLossV2_weight"] > 0:
            self.KullbackHistogramLossV2 = KullbackHistogramLossV2()

        if cfg["train"]["SalientRegionLoss_weight"] > 0:
            self.SalientRegionLoss = SalientRegionLoss()

        if cfg["train"]["glcmLoss_weight"] > 0:
            self.glcmLoss = glcmLoss()

        if cfg["train"]["GradientDomainLoss_weight"] > 0:
            self.GradientDomainLoss = GradientDomainLoss()

        if cfg["train"]["SobelLoss_weight"] > 0:
            self.SobelLoss = SobelLoss()

        if cfg["train"]["ColorHarmonyLoss_weight"] > 0:
            self.ColorHarmonyLoss = ColorHarmonyLoss()

        if cfg["train"]["VIT_FeatureLoss_weight"] > 0:
            self.VIT_FeatureLoss = VIT_FeatureLoss()

        if cfg["train"]["VIT_MMD_FeatureLoss_weight"] > 0:
            self.VIT_MMD_FeatureLoss = VIT_MMD_FeatureLoss()

        if cfg["train"]["TIMM_FeatureLoss_weight"] > 0:
            self.TIMM_FeatureLoss = TIMM_FeatureLoss(
                model_arch=cfg["train"]["TIMM_FeatureLoss_arch"],
                resolution=cfg["train"]["TIMM_FeatureLoss_resolution"],
                fp16=cfg["train"]["TIMM_FeatureLoss_fp16"],
                criterion=cfg["train"]["TIMM_FeatureLoss_criterion"],
                normalize=cfg["train"]["TIMM_FeatureLoss_normalize"],
                last_feature=cfg["train"]["TIMM_FeatureLoss_last_feature"],
            )

        if cfg["train"]["LaplacianLoss_weight"] > 0:
            self.LaplacianLoss = LaplacianLoss()

        if cfg["train"]["SobelLossV2_weight"] > 0:
            self.SobelLossV2 = SobelLossV2()

        if cfg["train"]["hrf_perceptual_weight"] > 0:
            from arch.hrf_perceptual import ResNetPL

            self.hrf_perceptual_loss = ResNetPL()
            for param in self.hrf_perceptual_loss.parameters():
                param.requires_grad = False

            if cfg["train"]["force_fp16_hrf"] is True:
                self.hrf_perceptual_loss = self.hrf_perceptual_loss.half()

        if cfg["train"]["YUVColorLoss_weight"] > 0:
            self.YUVColorLoss = YUVColorLoss()
        if cfg["train"]["XYZColorLoss_weight"] > 0:
            self.XYZColorLoss = XYZColorLoss()

        if cfg["train"]["FrobeniusNormLoss_weight"] > 0:
            self.FrobeniusNormLoss = FrobeniusNormLoss()
        if cfg["train"]["GradientLoss_weight"] > 0:
            self.GradientLoss = GradientLoss()
        if cfg["train"]["MultiscalePixelLoss_weight"] > 0:
            self.MultiscalePixelLoss = MultiscalePixelLoss()
        if cfg["train"]["SPLoss_weight"] > 0:
            self.SPLoss = SPLoss()

        # pytorch loss
        if cfg["train"]["Huber_weight"] > 0:
            self.HuberLoss = nn.HuberLoss()
        if cfg["train"]["SmoothL1_weight"] > 0:
            self.SmoothL1Loss = nn.SmoothL1Loss()
        if cfg["train"]["SoftMargin_weight"] > 0:
            self.SoftMarginLoss = nn.SoftMarginLoss()

        if cfg["train"]["Lap_weight"] > 0:
            self.LapLoss = LapLoss()

        if cfg["train"]["iqa_weight"] > 0:
            self.iqa_loss = IQA_loss()

        if cfg["network_G"]["netG"] == "rife":
            from loss.loss import SOBEL

            self.sobel = SOBEL()

        # discriminator loss
        if cfg["network_D"]["discriminator_criterion"] == "MSE":
            self.discriminator_criterion = torch.nn.MSELoss()

        # augmentation
        if cfg["train"]["augmentation_method"] == "MuarAugment":
            from loss.MuarAugment import BatchRandAugment, MuAugment

            rand_augment = BatchRandAugment(
                N_TFMS=cfg["train"]["N_TFMS"],
                MAGN=cfg["train"]["MAGN"],
                mean=[0.7032, 0.6346, 0.6234],
                std=[0.2520, 0.2507, 0.2417],
            )
            self.mu_transform = MuAugment(
                rand_augment,
                N_COMPS=cfg["train"]["N_COMPS"],
                N_SELECTED=cfg["train"]["N_SELECTED"],
            )
        elif cfg["train"]["augmentation_method"] == "batch_aug":
            from loss.batchaug import BatchAugment

            self.batch_aug = BatchAugment(
                mixopts=cfg["train"]["mixopts"],
                mixprob=cfg["train"]["mixprob"],
                mixalpha=cfg["train"]["mixalpha"],
                aux_mixprob=cfg["train"]["aux_mixprob"],
                aux_mixalpha=cfg["train"]["aux_mixalpha"],
            )
        elif cfg["train"]["augmentation_method"] == "diffaug":
            from loss.diffaug import DiffAugment

            self.DiffAugment = DiffAugment()

        self.cfg = cfg
        self.scaler = torch.cuda.amp.GradScaler()

    def forward(
        self,
        out,
        hr_image,
        writer=None,
        global_step=0,
        optimizer_idx=0,
        netD=None,
        other=None,
        other_teacher=None,
        log_suffix="",
    ):
        if self.cfg["network_D"]["netD"] is None:
            g_opt = self.optimizers()

        if self.cfg["network_D"]["netD"] is not None:
            g_opt, d_opt = self.optimizers()

        # train generator
        total_loss = 0
        if self.cfg["train"]["L1Loss_weight"] > 0:
            L1Loss_forward = self.cfg["train"]["L1Loss_weight"] * self.L1Loss(
                out, hr_image
            )
            total_loss += L1Loss_forward
            if self.cfg["logging"]:
                writer.add_scalar("loss/L1" + log_suffix, L1Loss_forward, global_step)

        if self.cfg["train"]["HFEN_weight"] > 0:
            HFENLoss_forward = self.cfg["train"]["HFEN_weight"] * self.HFENLoss(
                out, hr_image
            )
            total_loss += HFENLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/HFEN" + log_suffix, HFENLoss_forward, global_step
                )

        if self.cfg["train"]["Elastic_weight"] > 0:
            ElasticLoss_forward = self.cfg["train"][
                "Elastic_weight"
            ] * self.ElasticLoss(out, hr_image)
            total_loss += ElasticLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/Elastic" + log_suffix, ElasticLoss_forward, global_step
                )

        if self.cfg["train"]["Relative_l1_weight"] > 0:
            RelativeL1_forward = self.cfg["train"][
                "Relative_l1_weight"
            ] * self.RelativeL1(out, hr_image)
            total_loss += RelativeL1_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/RelativeL1" + log_suffix, RelativeL1_forward, global_step
                )

        if self.cfg["train"]["L1CosineSim_weight"] > 0:
            L1CosineSim_forward = self.cfg["train"][
                "L1CosineSim_weight"
            ] * self.L1CosineSim(out, hr_image)
            total_loss += L1CosineSim_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/L1CosineSim" + log_suffix, L1CosineSim_forward, global_step
                )

        if self.cfg["train"]["ClipL1_weight"] > 0:
            ClipL1_forward = self.cfg["train"]["ClipL1_weight"] * self.ClipL1(
                out, hr_image
            )
            total_loss += ClipL1_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/ClipL1" + log_suffix, ClipL1_forward, global_step
                )

        if self.cfg["train"]["FFTLoss_weight"] > 0:
            FFTloss_forward = self.cfg["train"]["FFTLoss_weight"] * self.FFTloss(
                out, hr_image
            )
            total_loss += FFTloss_forward
            if self.cfg["logging"]:
                writer.add_scalar("loss/FFT" + log_suffix, FFTloss_forward, global_step)

        if self.cfg["train"]["OFLoss_weight"] > 0:
            OFLoss_forward = self.cfg["train"]["OFLoss_weight"] * self.OFLoss(out)
            total_loss += OFLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar("loss/OF" + log_suffix, OFLoss_forward, global_step)

        if self.cfg["train"]["GPLoss_weight"] > 0:
            GPLoss_forward = self.cfg["train"]["GPLoss_weight"] * self.GPLoss(
                out, hr_image
            )
            total_loss += GPLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar("loss/GP" + log_suffix, GPLoss_forward, global_step)

        if self.cfg["train"]["CPLoss_weight"] > 0:
            CPLoss_forward = self.cfg["train"]["CPLoss_weight"] * self.CPLoss(
                out, hr_image
            )
            total_loss += CPLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar("loss/CP" + log_suffix, CPLoss_forward, global_step)

        if self.cfg["train"]["Contextual_weight"] > 0:
            Contextual_Loss_forward = self.cfg["train"][
                "Contextual_weight"
            ] * self.Contextual_Loss(out, hr_image)
            total_loss += Contextual_Loss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/contextual" + log_suffix, Contextual_Loss_forward, global_step
                )

        if self.cfg["train"]["StyleLoss_weight"] > 0:
            style_forward = self.cfg["train"]["StyleLoss_weight"] * self.StyleLoss(
                out, hr_image
            )
            total_loss += style_forward
            if self.cfg["logging"]:
                writer.add_scalar("loss/style" + log_suffix, style_forward, global_step)

        if self.cfg["train"]["TVLoss_weight"] > 0:
            tv_forward = self.cfg["train"]["TVLoss_weight"] * self.TVLoss(out)
            total_loss += tv_forward
            if self.cfg["logging"]:
                writer.add_scalar("loss/tv" + log_suffix, tv_forward, global_step)

        if self.cfg["train"]["perceptual_weight"] > 0:
            self.perceptual_loss.to(self.device)
            if (
                self.cfg["train"]["force_fp16_perceptual"] is True
                and self.cfg["train"]["perceptual_tensorrt"] is True
            ):
                perceptual_loss_forward = (
                    self.cfg["train"]["perceptual_weight"]
                    * self.perceptual_loss(out.half(), hr_image.half())[0]
                )
            elif (
                self.cfg["train"]["force_fp16_perceptual"] is True
                and self.cfg["train"]["perceptual_tensorrt"] is False
            ):
                perceptual_loss_forward = self.cfg["train"][
                    "perceptual_weight"
                ] * self.perceptual_loss(in0=out.half(), in1=hr_image.half())
            elif (
                self.cfg["train"]["force_fp16_perceptual"] is False
                and self.cfg["train"]["perceptual_tensorrt"] is True
            ):
                perceptual_loss_forward = (
                    self.cfg["train"]["perceptual_weight"]
                    * self.perceptual_loss(out, hr_image)[0]
                )
            elif (
                self.cfg["train"]["force_fp16_perceptual"] is False
                and self.cfg["train"]["perceptual_tensorrt"] is False
            ):
                perceptual_loss_forward = self.cfg["train"][
                    "perceptual_weight"
                ] * self.perceptual_loss(in0=out, in1=hr_image)
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/perceptual" + log_suffix,
                    perceptual_loss_forward.float(),
                    global_step,
                )
            total_loss += perceptual_loss_forward

        if self.cfg["train"]["hrf_perceptual_weight"] > 0:
            self.hrf_perceptual_loss.to(self.device)
            if self.cfg["train"]["force_fp16_hrf"] is True:
                hrf_perceptual_loss_forward = self.cfg["train"][
                    "hrf_perceptual_weight"
                ] * self.hrf_perceptual_loss(out.half(), hr_image.half())
            else:
                hrf_perceptual_loss_forward = self.cfg["train"][
                    "hrf_perceptual_weight"
                ] * self.hrf_perceptual_loss(out, hr_image)
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/hrf_perceptual" + log_suffix,
                    hrf_perceptual_loss_forward,
                    global_step,
                )
            total_loss += hrf_perceptual_loss_forward

        if self.cfg["train"]["MSE_weight"] > 0:
            MSE_forward = self.cfg["train"]["MSE_weight"] * self.MSELoss(out, hr_image)
            total_loss += MSE_forward
            if self.cfg["logging"]:
                writer.add_scalar("loss/MSE" + log_suffix, MSE_forward, global_step)

        if self.cfg["train"]["BCE_weight"] > 0:
            BCELogits_forward = self.cfg["train"]["BCE_weight"] * self.BCELogits(
                out, hr_image
            )
            total_loss += BCELogits_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/BCELogits" + log_suffix, BCELogits_forward, global_step
                )

        if self.cfg["train"]["Huber_weight"] > 0:
            Huber_forward = self.cfg["train"]["Huber_weight"] * self.HuberLoss(
                out, hr_image
            )
            total_loss += Huber_forward
            if self.cfg["logging"]:
                writer.add_scalar("loss/Huber" + log_suffix, Huber_forward, global_step)

        if self.cfg["train"]["SmoothL1_weight"] > 0:
            SmoothL1_forward = self.cfg["train"]["SmoothL1_weight"] * self.SmoothL1Loss(
                out, hr_image
            )
            total_loss += SmoothL1_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/SmoothL1" + log_suffix, SmoothL1_forward, global_step
                )

        if self.cfg["train"]["SoftMargin_weight"] > 0:
            SoftMargin_forward = self.cfg["train"][
                "SoftMargin_weight"
            ] * self.SoftMarginLoss(out, hr_image)
            total_loss += SoftMargin_forward

            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/SoftMargin" + log_suffix, SoftMargin_forward, global_step
                )

        if self.cfg["train"]["Lap_weight"] > 0:
            Lap_forward = (
                self.cfg["train"]["Lap_weight"] * (self.LapLoss(out, hr_image)).mean()
            )
            total_loss += Lap_forward
            if self.cfg["logging"]:
                writer.add_scalar("loss/Lap", Lap_forward, global_step)

        if self.cfg["train"]["YUVColorLoss_weight"] > 0:
            YUVColorLoss_forward = self.cfg["train"]["YUVColorLoss_weight"] * (
                self.YUVColorLoss(out, hr_image)
            )
            total_loss += YUVColorLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/YUVColorLoss" + log_suffix, YUVColorLoss_forward, global_step
                )

        if self.cfg["train"]["XYZColorLoss_weight"] > 0:
            XYZColorLoss_forward = self.cfg["train"]["XYZColorLoss_weight"] * (
                self.XYZColorLoss(out, hr_image)
            )
            total_loss += XYZColorLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/XYZColorLoss" + log_suffix, XYZColorLoss_forward, global_step
                )

        if self.cfg["train"]["FrobeniusNormLoss_weight"] > 0:
            FrobeniusNormLoss_forward = self.cfg["train"][
                "FrobeniusNormLoss_weight"
            ] * self.FrobeniusNormLoss(out, hr_image)
            total_loss += FrobeniusNormLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/FrobeniusNormLoss" + log_suffix,
                    FrobeniusNormLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["GradientLoss_weight"] > 0:
            GradientLoss_forward = self.cfg["train"][
                "GradientLoss_weight"
            ] * self.GradientLoss(out, hr_image)
            total_loss += GradientLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/GradientLoss" + log_suffix, GradientLoss_forward, global_step
                )

        if self.cfg["train"]["MultiscalePixelLoss_weight"] > 0:
            MultiscalePixelLoss_forward = self.cfg["train"][
                "MultiscalePixelLoss_weight"
            ] * self.MultiscalePixelLoss(out, hr_image)
            total_loss += MultiscalePixelLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/MultiscalePixelLoss" + log_suffix,
                    MultiscalePixelLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["SPLoss_weight"] > 0:
            SPLoss_forward = self.cfg["train"]["SPLoss_weight"] * (
                self.SPLoss(out, hr_image)
            )
            total_loss += SPLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/SPLoss" + log_suffix, SPLoss_forward, global_step
                )

        if self.cfg["train"]["FFLoss_weight"] > 0:
            FFLoss_forward = self.cfg["train"]["FFLoss_weight"] * self.FFLoss(
                out.type(torch.cuda.FloatTensor),
                hr_image.type(torch.cuda.FloatTensor),
            )
            total_loss += FFLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/FFLoss" + log_suffix, FFLoss_forward, global_step
                )

        if self.cfg["train"]["Canny_weight"] > 0:
            Canny_forward = self.cfg["train"]["Canny_weight"] * self.CannyLoss(
                out, hr_image
            )
            total_loss += Canny_forward
            if self.cfg["logging"]:
                writer.add_scalar("loss/Canny" + log_suffix, Canny_forward, global_step)

        # experimental loss

        if self.cfg["train"]["KullbackHistogramLoss_weight"] > 0:
            KullbackHistogramLoss_forward = self.cfg["train"][
                "KullbackHistogramLoss_weight"
            ] * self.KullbackHistogramLoss(out.float(), hr_image.float())
            total_loss += KullbackHistogramLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/KullbackHistogramLoss" + log_suffix,
                    KullbackHistogramLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["KullbackHistogramLossV2_weight"] > 0:
            KullbackHistogramLoss_forward = self.cfg["train"][
                "KullbackHistogramLossV2_weight"
            ] * self.KullbackHistogramLossV2(out.float(), hr_image.float())
            total_loss += KullbackHistogramLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/KullbackHistogramLossV2" + log_suffix,
                    KullbackHistogramLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["SalientRegionLoss_weight"] > 0:
            SalientRegionLoss_forward = self.cfg["train"][
                "SalientRegionLoss_weight"
            ] * self.SalientRegionLoss(out.float(), hr_image.float())
            total_loss += SalientRegionLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/SalientRegionLoss" + log_suffix,
                    SalientRegionLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["glcmLoss_weight"] > 0:
            glcmLoss_forward = self.cfg["train"]["glcmLoss_weight"] * self.glcmLoss(
                out, hr_image
            )
            total_loss += glcmLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/glcm" + log_suffix,
                    glcmLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["GradientDomainLoss_weight"] > 0:
            GradientDomainLoss_forward = self.cfg["train"][
                "GradientDomainLoss_weight"
            ] * self.GradientDomainLoss(out, hr_image)
            total_loss += GradientDomainLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/GradientDomainLoss" + log_suffix,
                    GradientDomainLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["SobelLoss_weight"] > 0:
            SobelLoss_forward = self.cfg["train"]["SobelLoss_weight"] * self.SobelLoss(
                out, hr_image
            )
            total_loss += SobelLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/SobelLoss" + log_suffix,
                    SobelLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["ColorHarmonyLoss_weight"] > 0:
            ColorHarmonyLoss_forward = self.cfg["train"][
                "ColorHarmonyLoss_weight"
            ] * self.ColorHarmonyLoss(out, hr_image)
            total_loss += ColorHarmonyLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/ColorHarmonyLoss" + log_suffix,
                    ColorHarmonyLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["VIT_FeatureLoss_weight"] > 0:
            VIT_FeatureLoss_forward = self.cfg["train"][
                "VIT_FeatureLoss_weight"
            ] * self.VIT_FeatureLoss(out, hr_image)
            total_loss += VIT_FeatureLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/VIT_FeatureLoss" + log_suffix,
                    VIT_FeatureLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["VIT_MMD_FeatureLoss_weight"] > 0:
            VIT_MMD_FeatureLoss_forward = self.cfg["train"][
                "VIT_MMD_FeatureLoss_weight"
            ] * self.VIT_MMD_FeatureLoss(out, hr_image)
            total_loss += VIT_MMD_FeatureLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/VIT_MMD_FeatureLoss" + log_suffix,
                    VIT_MMD_FeatureLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["TIMM_FeatureLoss_weight"] > 0:
            TIMM_FeatureLoss_forward = self.cfg["train"][
                "TIMM_FeatureLoss_weight"
            ] * self.TIMM_FeatureLoss(out, hr_image)
            total_loss += TIMM_FeatureLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/TIMM_FeatureLoss" + log_suffix,
                    TIMM_FeatureLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["LaplacianLoss_weight"] > 0:
            LaplacianLoss_forward = self.cfg["train"][
                "LaplacianLoss_weight"
            ] * self.LaplacianLoss(out, hr_image)
            total_loss += LaplacianLoss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/LaplacianLoss" + log_suffix,
                    LaplacianLoss_forward,
                    global_step,
                )

        if self.cfg["train"]["SobelLossV2_weight"] > 0:
            SobelLossV2_forward = self.cfg["train"][
                "SobelLossV2_weight"
            ] * self.SobelLossV2(out, hr_image)
            total_loss += SobelLossV2_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/SobelLossV2" + log_suffix,
                    SobelLossV2_forward,
                    global_step,
                )

        if self.cfg["train"]["textured_loss_weight"] > 0:
            textured_loss_forward = self.cfg["train"][
                "textured_loss_weight"
            ] * self.textured_loss(out, hr_image)
            total_loss += textured_loss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/textured_loss" + log_suffix,
                    textured_loss_forward,
                    global_step,
                )

        if self.cfg["train"]["ldl_weight"] > 0:
            ldl_loss_forward = self.cfg["train"]["ldl_weight"] * self.ldl_loss(
                out, hr_image
            )
            total_loss += ldl_loss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/ldl" + log_suffix, ldl_loss_forward, global_step
                )

        if self.cfg["train"]["iqa_weight"] > 0:
            iqa_loss_forward = self.cfg["train"]["iqa_weight"] * self.iqa_loss(
                out, hr_image
            )
            total_loss += iqa_loss_forward
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/iqa_" + self.cfg["train"]["iqa_metric"] + log_suffix,
                    iqa_loss_forward,
                    global_step,
                )

        #########################
        # exotic loss
        # if model has two output, also calculate loss for such an image
        # example with just l1 loss
        if self.cfg["network_G"]["netG"] in (
            "deepfillv1",
            "deepfillv2",
            "Adaptive",
        ):
            l1_stage1 = self.cfg["train"]["stage1_weight"] * self.L1Loss(
                other["other_img"], hr_image
            )
            total_loss += l1_stage1
            if self.cfg["logging"]:
                writer.add_scalar("loss/l1_stage1" + log_suffix, l1_stage1, global_step)

        # CSA Loss
        if self.cfg["network_G"]["netG"] == "CSA":
            recon_loss = self.L1Loss(other["coarse_result"], hr_image) + self.L1Loss(
                out, hr_image
            )
            cons = self.ConsistencyLoss()
            cons_loss = cons(other["csa"], other["csa_d"], hr_image, other["mask"])

            total_loss += recon_loss
            total_loss += cons_loss

            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/recon_loss" + log_suffix, recon_loss, global_step
                )
                writer.add_scalar("loss/cons_loss" + log_suffix, cons_loss, global_step)

        # EdgeConnect
        if self.cfg["network_G"]["netG"] == "EdgeConnect":
            l1_edge = self.L1Loss(other["other_img"], other["edge"])
            total_loss += l1_edge
            if self.cfg["logging"]:
                writer.add_scalar("loss/l1_edge" + log_suffix, l1_edge, global_step)

        # PVRS
        if self.cfg["network_G"]["netG"] == "PVRS":
            edge_big_l1 = self.L1Loss(other["edge_big"], other["edge"])
            edge_small_l1 = self.L1Loss(
                other["edge_small"],
                torch.nn.functional.interpolate(other["edge"], scale_factor=0.5),
            )
            total_loss += edge_big_l1
            total_loss += edge_small_l1

            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/edge_big_l1" + log_suffix, edge_big_l1, global_step
                )
                writer.add_scalar(
                    "loss/edge_small_l1" + log_suffix, edge_small_l1, global_step
                )

        # FRRN
        if self.cfg["network_G"]["netG"] == "FRRN":
            mid_l1_loss = 0
            for idx in range(len(other["mid_x"]) - 1):
                mid_l1_loss += self.L1Loss(
                    other["mid_x"][idx] * other["mid_mask"][idx],
                    hr_image * other["mid_mask"][idx],
                )
            total_loss += mid_l1_loss
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/mid_l1_loss" + log_suffix, mid_l1_loss, global_step
                )

        if self.cfg["logging"]:
            writer.add_scalar("loss/g_loss" + log_suffix, total_loss, global_step)

        # srflow
        """
        # todo
        if self.cfg["network_G"]["netG"] == "srflow":
            nll_loss = torch.mean(nll)
            total_loss += self.cfg["network_G"]["nll_weight"] * nll_loss
            writer.add_scalar("loss/nll_loss" + log_suffix, nll_loss, global_step)
        """

        # CTSDG
        if self.cfg["network_G"]["netG"] == "CTSDG":
            edge_loss = (
                self.BCE(other["projected_edge"], other["edge"])
                * self.cfg["train"]["CTSDG_edge_weight"]
            )
            total_loss += edge_loss
            if self.cfg["logging"]:
                writer.add_scalar("loss/edge_loss" + log_suffix, edge_loss, global_step)

            projected_loss = (
                self.L1Loss(other["projected_image"], hr_image)
                * self.cfg["train"]["CTSDG_projected_weight"]
            )
            total_loss += projected_loss
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/projected_loss" + log_suffix, projected_loss, global_step
                )

        # rife
        if self.cfg["network_G"]["netG"] == "rife":
            sobel_loss = (
                self.sobel(other["flow"][3], other["flow"][3] * 0).mean()
                * self.cfg["train"]["SOBEL_weight"]
            )
            total_loss += sobel_loss
            if self.cfg["logging"]:
                writer.add_scalar(
                    "loss/sobel_loss" + log_suffix, sobel_loss, global_step
                )

        # feature match loss for knowledge destillation
        if (
            self.cfg["network_G_teacher"]["netG"] in ("MRRDBNet_FM", "SRVGGNetCompact")
            and log_suffix == "_teacher"
        ):
            fm_loss = 0
            for i, fm in enumerate(other["feature_maps"]):
                # sometimes the features dont exactly match, to avoid crashing, a check
                if fm.shape == other_teacher["feature_maps"][i].shape:
                    fm_loss += self.l1(fm, other_teacher["feature_maps"][i])
            fm_loss = fm_loss * self.cfg["network_G_teacher"]["l1_feature_maps_weight"]
            if self.cfg["logging"]:
                writer.add_scalar("loss/fm_loss" + log_suffix, fm_loss, global_step)

        ##########################
        # discriminator loss
        ##########################

        # replicating realesrgan strategy, using normal gt for discriminator
        if self.cfg["datasets"]["train"]["mode"] == "DS_realesrgan":
            hr_image = other["gt"]

        if self.cfg["network_D"]["netD"] is not None:
            # Try to fool the discriminator
            Tensor = torch.FloatTensor
            fake = (
                Variable(Tensor((out.shape[0])).fill_(0.0), requires_grad=False)
                .unsqueeze(-1)
                .to(self.device)
            )

            if self.cfg["network_D"]["netD"] == "resnet3d":
                # 3d
                if self.cfg["train"]["augmentation_method"] == "diffaug":
                    d_loss_fool = self.cfg["network_D"][
                        "d_loss_fool_weight"
                    ] * self.discriminator_criterion(
                        netD(
                            self.DiffAugment(
                                torch.stack(
                                    [other["hr_image1"], out, other["hr_image3"]],
                                    dim=1,
                                ),
                                self.cfg["train"]["policy"],
                            )
                        ),
                        fake,
                    )
                else:
                    d_loss_fool = self.cfg["network_D"][
                        "d_loss_fool_weight"
                    ] * self.discriminator_criterion(
                        netD(
                            torch.stack(
                                [other["hr_image1"], out, other["hr_image3"]], dim=1
                            )
                        ),
                        fake,
                    )
            else:
                # 2d
                if self.cfg["train"]["augmentation_method"] == "diffaug":
                    d_loss_fool = self.cfg["network_D"][
                        "d_loss_fool_weight"
                    ] * self.discriminator_criterion(
                        netD(self.DiffAugment(out, self.cfg["train"]["policy"])),
                        fake,
                    )
                elif self.cfg["train"]["augmentation_method"] == "MuarAugment":
                    self.mu_transform.setup(self)
                    mu_augment, _ = self.mu_transform((out, fake))
                    d_loss_fool = self.cfg["network_D"][
                        "d_loss_fool_weight"
                    ] * self.discriminator_criterion(
                        netD(mu_augment).float(), fake.float()
                    )
                else:
                    if self.cfg["network_D"]["netD"] == "FFCNLayerDiscriminator":
                        FFCN_class, FFCN_feature = netD(out)
                        d_loss_fool = self.cfg["network_D"][
                            "d_loss_fool_weight"
                        ] * self.discriminator_criterion(FFCN_class, fake)
                    else:
                        d_loss_fool = self.cfg["network_D"][
                            "d_loss_fool_weight"
                        ] * self.discriminator_criterion(netD(out), fake)

                total_loss += d_loss_fool
                if self.cfg["logging"]:
                    writer.add_scalar(
                        "loss/d_loss_fool" + log_suffix, d_loss_fool, global_step
                    )

                if (
                    self.cfg["network_D"]["netD"] == "FFCNLayerDiscriminator"
                    and self.cfg["network_D"]["FFCN_feature_weight"] > 0
                ):
                    FFCN_class_orig, FFCN_feature_orig = netD(hr_image)
                    # dont give mask if it's not available
                    if self.cfg["network_G"]["netG"] in (
                        "CDFI",
                        "sepconv_enhanced",
                        "CAIN",
                        "rife",
                        "RRIN",
                        "ABME",
                        "EDSC",
                    ):
                        feature_matching_loss_forward = self.cfg["network_D"][
                            "FFCN_feature_weight"
                        ] * feature_matching_loss(FFCN_feature, FFCN_feature_orig)
                    else:
                        feature_matching_loss_forward = self.cfg["network_D"][
                            "FFCN_feature_weight"
                        ] * feature_matching_loss(
                            FFCN_feature, FFCN_feature_orig, hr_image
                        )
                    total_loss += feature_matching_loss_forward
                    if self.cfg["logging"]:
                        writer.add_scalar(
                            "loss/feature_matching_loss" + log_suffix,
                            feature_matching_loss_forward,
                            global_step,
                        )

        # optimizer
        self.toggle_optimizer(g_opt)
        g_opt.zero_grad()
        # self.manual_backward(total_loss, retain_graph=True)
        if self.cfg["train"]["gradient_clipping_G"]:
            self.clip_gradients(
                g_opt,
                gradient_clip_val=self.cfg["train"]["gradient_clipping_G_value"],
                gradient_clip_algorithm="norm",
            )

        self.scaler.scale(total_loss).backward()
        self.scaler.step(g_opt)
        self.scaler.update()

        # g_opt.step()
        self.untoggle_optimizer(g_opt)

        if self.cfg["network_D"]["netD"] is None:
            return total_loss

        ##########################
        # train discriminator
        ##########################

        hr_image = hr_image.clone().detach()
        out = out.clone().detach()
        self.toggle_optimizer(d_opt)

        # replicating realesrgan strategy, using normal gt for discriminator
        if self.cfg["datasets"]["train"]["mode"] == "DS_realesrgan":
            hr_image = other["gt"]

        if self.cfg["network_D"]["netD"] is not None:
            Tensor = torch.FloatTensor  # if cuda else torch.FloatTensor
            valid = (
                Variable(Tensor((out.shape[0])).fill_(1.0), requires_grad=False)
                .unsqueeze(-1)
                .to(self.device)
            )
            fake = (
                Variable(Tensor((out.shape[0])).fill_(0.0), requires_grad=False)
                .unsqueeze(-1)
                .to(self.device)
            )

            if self.cfg["network_D"]["netD"] == "resnet3d":
                # 3d
                if self.cfg["train"]["augmentation_method"] == "diffaug":
                    dis_real_loss = self.discriminator_criterion(
                        netD(
                            self.DiffAugment(
                                torch.stack(
                                    [
                                        other["hr_image1"],
                                        hr_image,
                                        other["hr_image3"],
                                    ],
                                    dim=1,
                                ),
                                self.cfg["train"]["policy"],
                            )
                        ),
                        valid,
                    )
                    dis_fake_loss = self.discriminator_criterion(
                        netD(
                            torch.stack(
                                [other["hr_image1"], out, other["hr_image3"]], dim=1
                            )
                        ),
                        fake,
                    )
                else:
                    dis_real_loss = self.discriminator_criterion(
                        netD(
                            torch.stack(
                                [other["hr_image1"], hr_image, other["hr_image3"]],
                                dim=1,
                            )
                        ),
                        valid,
                    )
                    dis_fake_loss = self.discriminator_criterion(
                        netD(
                            torch.stack(
                                [other["hr_image1"], out, other["hr_image3"]], dim=1
                            )
                        ),
                        fake,
                    )
            else:
                # 2d
                if self.cfg["train"]["augmentation_method"] == "diffaug":
                    discr_out_fake = netD(
                        self.DiffAugment(out, self.cfg["train"]["policy"])
                    )
                    discr_out_real = netD(
                        self.DiffAugment(hr_image, self.cfg["train"]["policy"])
                    )
                elif self.cfg["train"]["augmentation_method"] == "MuarAugment":
                    self.mu_transform.setup(self)
                    # fake
                    mu_augment, _ = self.mu_transform((out, fake))
                    discr_out_fake = netD(mu_augment)
                    # real
                    mu_augment, _ = self.mu_transform((hr_image, valid))
                    discr_out_real = netD(mu_augment)
                elif self.cfg["train"]["augmentation_method"] == "batch_aug":
                    fake_out, real_out = self.batch_aug(out, hr_image)
                    discr_out_fake = netD(fake_out)
                    discr_out_real = netD(real_out)
                else:
                    if self.cfg["network_D"]["netD"] == "FFCNLayerDiscriminator":
                        discr_out_fake, _ = netD(out)
                        discr_out_real, _ = netD(hr_image)
                    else:
                        discr_out_fake = netD(out)
                        discr_out_real = netD(hr_image)

                dis_fake_loss = self.discriminator_criterion(discr_out_fake, fake)
                dis_real_loss = self.discriminator_criterion(discr_out_real, fake)

            # Total loss
            d_loss = self.cfg["network_D"]["d_loss_weight"] * (
                (dis_real_loss + dis_fake_loss) / 2
            )

            if self.cfg["logging"]:
                writer.add_scalar("loss/d_loss" + log_suffix, d_loss, global_step)

            # todo: scale optimizer
            d_opt.zero_grad()
            self.manual_backward(d_loss)
            if self.cfg["train"]["gradient_clipping_D"]:
                self.clip_gradients(
                    d_opt,
                    gradient_clip_val=self.cfg["train"]["gradient_clipping_D_value"],
                    gradient_clip_algorithm="norm",
                )
            d_opt.step()
            self.untoggle_optimizer(d_opt)
        return total_loss
