import pytorch_lightning as pl
from loss.loss import (
    FocalFrequencyLoss,
    feature_matching_loss,
    FrobeniusNormLoss,
    LapLoss,
    CharbonnierLoss,
    GANLoss,
    GradientPenaltyLoss,
    HFENLoss,
    TVLoss,
    GradientLoss,
    ElasticLoss,
    RelativeL1,
    L1CosineSim,
    ClipL1,
    MaskedL1Loss,
    MultiscalePixelLoss,
    FFTloss,
    OFLoss,
    L1_regularization,
    YUVColorLoss,
    XYZColorLoss,
    AverageLoss,
    GPLoss,
    CPLoss,
    SPL_ComputeWithTrace,
    SPLoss,
    Contextual_Loss,
    StyleLoss,
    ConsistencyLoss,
    CannyLoss,
)
from piq import (
    SSIMLoss,
    MultiScaleSSIMLoss,
    VIFLoss,
    FSIMLoss,
    GMSDLoss,
    MultiScaleGMSDLoss,
    VSILoss,
    HaarPSILoss,
    MDSILoss,
    BRISQUELoss,
    PieAPP,
    DISTS,
    IS,
    FID,
    KID,
    PR,
)
import torch
import torch.nn as nn
import os
from torch.autograd import Variable


class AllLoss(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False

        # loss functions
        self.l1 = nn.L1Loss()

        if cfg["train"]["loss_f"] == "L1Loss":
            loss_f = torch.nn.L1Loss()
        elif cfg["train"]["loss_f"] == "L1CosineSim":
            loss_f = L1CosineSim(
                loss_lambda=cfg["train"]["loss_lambda"],
                reduction=cfg["train"]["reduction_L1CosineSim"],
            )

        self.HFENLoss = HFENLoss(
            loss_f=loss_f,
            kernel=cfg["train"]["kernel"],
            kernel_size=cfg["train"]["kernel_size"],
            sigma=cfg["train"]["sigma"],
            norm=cfg["train"]["norm"],
        )
        self.ElasticLoss = ElasticLoss(
            a=cfg["train"]["a"], reduction=cfg["train"]["reduction_elastic"]
        )
        self.RelativeL1 = RelativeL1(
            eps=cfg["train"]["l1_eps"], reduction=cfg["train"]["reduction_relative"]
        )
        self.L1CosineSim = L1CosineSim(
            loss_lambda=cfg["train"]["loss_lambda"],
            reduction=cfg["train"]["reduction_L1CosineSim"],
        )
        self.ClipL1 = ClipL1(
            clip_min=cfg["train"]["clip_min"], clip_max=cfg["train"]["clip_max"]
        )

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
        self.OFLoss = OFLoss()
        self.GPLoss = GPLoss(
            trace=cfg["train"]["gp_trace"], spl_denorm=cfg["train"]["gp_spl_denorm"]
        )
        self.CPLoss = CPLoss(
            rgb=cfg["train"]["rgb"],
            yuv=cfg["train"]["yuv"],
            yuvgrad=cfg["train"]["yuvgrad"],
            trace=cfg["train"]["cp_trace"],
            spl_denorm=cfg["train"]["cp_spl_denorm"],
            yuv_denorm=cfg["train"]["yuv_denorm"],
        )
        self.StyleLoss = StyleLoss()
        self.TVLoss = TVLoss(tv_type=cfg["train"]["tv_type"], p=cfg["train"]["p"])
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

        self.MSELoss = torch.nn.MSELoss()
        self.L1Loss = nn.L1Loss()
        self.BCELogits = torch.nn.BCEWithLogitsLoss()
        self.BCE = torch.nn.BCELoss()
        self.FFLoss = FocalFrequencyLoss()

        # perceptual loss
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
            torch.load(model_path, map_location=torch.device(self.device)), strict=False
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

        self.ConsistencyLoss = ConsistencyLoss()

        self.CannyLoss = CannyLoss(
            threshold=cfg["train"]["canny_threshold"],
            blurred_img_weight=cfg["train"]["canny_blurred_img_weight"],
            grad_mag_weight=cfg["train"]["canny_grad_mag_weight"],
            grad_orientation_weight=cfg["train"]["canny_grad_mag_weight"],
            thin_edges_weight=cfg["train"]["canny_thin_edges_weight"],
            thresholded_weight=cfg["train"]["canny_thresholded_weight"],
            early_threshold=cfg["train"]["canny_early_threshold"],
        )

        from arch.hrf_perceptual import ResNetPL

        self.hrf_perceptual_loss = ResNetPL()
        for param in self.hrf_perceptual_loss.parameters():
            param.requires_grad = False

        if cfg["train"]["force_fp16_hrf"] is True:
            self.hrf_perceptual_loss = self.hrf_perceptual_loss.half()

        self.YUVColorLoss = YUVColorLoss()
        self.XYZColorLoss = XYZColorLoss()

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

    def forward(
        self,
        out,
        hr_image,
        writer,
        global_step,
        optimizer_idx,
        netD,
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
            writer.add_scalar("loss/L1" + log_suffix, L1Loss_forward, global_step)

        if self.cfg["train"]["HFEN_weight"] > 0:
            HFENLoss_forward = self.cfg["train"]["HFEN_weight"] * self.HFENLoss(
                out, hr_image
            )
            total_loss += HFENLoss_forward
            writer.add_scalar("loss/HFEN" + log_suffix, HFENLoss_forward, global_step)

        if self.cfg["train"]["Elastic_weight"] > 0:
            ElasticLoss_forward = self.cfg["train"][
                "Elastic_weight"
            ] * self.ElasticLoss(out, hr_image)
            total_loss += ElasticLoss_forward
            writer.add_scalar(
                "loss/Elastic" + log_suffix, ElasticLoss_forward, global_step
            )

        if self.cfg["train"]["Relative_l1_weight"] > 0:
            RelativeL1_forward = self.cfg["train"][
                "Relative_l1_weight"
            ] * self.RelativeL1(out, hr_image)
            total_loss += RelativeL1_forward
            writer.add_scalar(
                "loss/RelativeL1" + log_suffix, RelativeL1_forward, global_step
            )

        if self.cfg["train"]["L1CosineSim_weight"] > 0:
            L1CosineSim_forward = self.cfg["train"][
                "L1CosineSim_weight"
            ] * self.L1CosineSim(out, hr_image)
            total_loss += L1CosineSim_forward
            writer.add_scalar(
                "loss/L1CosineSim" + log_suffix, L1CosineSim_forward, global_step
            )

        if self.cfg["train"]["ClipL1_weight"] > 0:
            ClipL1_forward = self.cfg["train"]["ClipL1_weight"] * self.ClipL1(
                out, hr_image
            )
            total_loss += ClipL1_forward
            writer.add_scalar("loss/ClipL1" + log_suffix, ClipL1_forward, global_step)

        if self.cfg["train"]["FFTLoss_weight"] > 0:
            FFTloss_forward = self.cfg["train"]["FFTLoss_weight"] * self.FFTloss(
                out, hr_image
            )
            total_loss += FFTloss_forward
            writer.add_scalar("loss/FFT" + log_suffix, FFTloss_forward, global_step)

        if self.cfg["train"]["OFLoss_weight"] > 0:
            OFLoss_forward = self.cfg["train"]["OFLoss_weight"] * self.OFLoss(out)
            total_loss += OFLoss_forward
            writer.add_scalar("loss/OF" + log_suffix, OFLoss_forward, global_step)

        if self.cfg["train"]["GPLoss_weight"] > 0:
            GPLoss_forward = self.cfg["train"]["GPLoss_weight"] * self.GPLoss(
                out, hr_image
            )
            total_loss += GPLoss_forward
            writer.add_scalar("loss/GP" + log_suffix, GPLoss_forward, global_step)

        if self.cfg["train"]["CPLoss_weight"] > 0:
            CPLoss_forward = self.cfg["train"]["CPLoss_weight"] * self.CPLoss(
                out, hr_image
            )
            total_loss += CPLoss_forward
            writer.add_scalar("loss/CP" + log_suffix, CPLoss_forward, global_step)

        if self.cfg["train"]["Contextual_weight"] > 0:
            Contextual_Loss_forward = self.cfg["train"][
                "Contextual_weight"
            ] * self.Contextual_Loss(out, hr_image)
            total_loss += Contextual_Loss_forward
            writer.add_scalar(
                "loss/contextual" + log_suffix, Contextual_Loss_forward, global_step
            )

        if self.cfg["train"]["StyleLoss_weight"] > 0:
            style_forward = self.cfg["train"]["StyleLoss_weight"] * self.StyleLoss(
                out, hr_image
            )
            total_loss += style_forward
            writer.add_scalar("loss/style" + log_suffix, style_forward, global_step)

        if self.cfg["train"]["TVLoss_weight"] > 0:
            tv_forward = self.cfg["train"]["TVLoss_weight"] * self.TVLoss(out)
            total_loss += tv_forward
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
            writer.add_scalar(
                "loss/perceptual" + log_suffix, perceptual_loss_forward, global_step
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
            writer.add_scalar(
                "loss/hrf_perceptual" + log_suffix,
                hrf_perceptual_loss_forward,
                global_step,
            )
            total_loss += hrf_perceptual_loss_forward

        if self.cfg["train"]["MSE_weight"] > 0:
            MSE_forward = self.cfg["train"]["MSE_weight"] * self.MSELoss(out, hr_image)
            total_loss += MSE_forward
            writer.add_scalar("loss/MSE" + log_suffix, MSE_forward, global_step)

        if self.cfg["train"]["BCE_weight"] > 0:
            BCELogits_forward = self.cfg["train"]["BCE_weight"] * self.BCELogits(
                out, hr_image
            )
            total_loss += BCELogits_forward
            writer.add_scalar(
                "loss/BCELogits" + log_suffix, BCELogits_forward, global_step
            )

        if self.cfg["train"]["Huber_weight"] > 0:
            Huber_forward = self.cfg["train"]["Huber_weight"] * self.HuberLoss(
                out, hr_image
            )
            total_loss += Huber_forward
            writer.add_scalar("loss/Huber" + log_suffix, Huber_forward, global_step)

        if self.cfg["train"]["SmoothL1_weight"] > 0:
            SmoothL1_forward = self.cfg["train"]["SmoothL1_weight"] * self.SmoothL1Loss(
                out, hr_image
            )
            total_loss += SmoothL1_forward
            writer.add_scalar(
                "loss/SmoothL1" + log_suffix, SmoothL1_forward, global_step
            )

        if self.cfg["train"]["Lap_weight"] > 0:
            Lap_forward = (
                self.cfg["train"]["Lap_weight"] * (self.LapLoss(out, hr_image)).mean()
            )
            total_loss += Lap_forward
            writer.add_scalar("loss/Lap", Lap_forward, global_step)

        if self.cfg["train"]["YUVColorLoss_weight"] > 0:
            YUVColorLoss_forward = self.cfg["train"]["YUVColorLoss_weight"] * (
                self.YUVColorLoss(out, hr_image)
            )
            total_loss += YUVColorLoss_forward
            writer.add_scalar(
                "loss/YUVColorLoss" + log_suffix, YUVColorLoss_forward, global_step
            )

        if self.cfg["train"]["XYZColorLoss_weight"] > 0:
            XYZColorLoss_forward = self.cfg["train"]["XYZColorLoss_weight"] * (
                self.XYZColorLoss(out, hr_image)
            )
            total_loss += XYZColorLoss_forward
            writer.add_scalar(
                "loss/XYZColorLoss" + log_suffix, XYZColorLoss_forward, global_step
            )

        if self.cfg["train"]["FrobeniusNormLoss_weight"] > 0:
            FrobeniusNormLoss_forward = self.cfg["train"][
                "FrobeniusNormLoss_weight"
            ] * self.FrobeniusNormLoss(out, hr_image)
            total_loss += FrobeniusNormLoss_forward
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
            writer.add_scalar(
                "loss/GradientLoss" + log_suffix, GradientLoss_forward, global_step
            )

        if self.cfg["train"]["MultiscalePixelLoss_weight"] > 0:
            MultiscalePixelLoss_forward = self.cfg["train"][
                "MultiscalePixelLoss_weight"
            ] * self.MultiscalePixelLoss(out, hr_image)
            total_loss += MultiscalePixelLoss_forward
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
            writer.add_scalar("loss/SPLoss" + log_suffix, SPLoss_forward, global_step)

        if self.cfg["train"]["FFLoss_weight"] > 0:
            FFLoss_forward = self.cfg["train"]["FFLoss_weight"] * self.FFLoss(
                out.type(torch.cuda.FloatTensor),
                hr_image.type(torch.cuda.FloatTensor),
            )
            total_loss += FFLoss_forward
            writer.add_scalar("loss/FFLoss" + log_suffix, FFLoss_forward, global_step)

        if self.cfg["train"]["SSIMLoss_weight"] > 0:
            SSIMLoss_forward = self.cfg["train"]["SSIMLoss_weight"] * self.SSIMLoss(
                out, hr_image
            )
            total_loss += SSIMLoss_forward
            writer.add_scalar("loss/SSIM" + log_suffix, SSIMLoss_forward, global_step)

        if self.cfg["train"]["MultiScaleSSIMLoss_weight"] > 0:
            MultiScaleSSIMLoss_forward = self.cfg["train"][
                "MultiScaleSSIMLoss_weight"
            ] * (self.MultiScaleSSIMLoss(out, hr_image))
            total_loss += MultiScaleSSIMLoss_forward
            writer.add_scalar(
                "loss/MultiScaleSSIM" + log_suffix,
                MultiScaleSSIMLoss_forward,
                global_step,
            )

        if self.cfg["train"]["VIFLoss_weight"] > 0:
            VIFLoss_forward = self.cfg["train"]["VIFLoss_weight"] * self.VIFLoss(
                out, hr_image
            )
            total_loss += VIFLoss_forward
            writer.add_scalar("loss/VIF" + log_suffix, VIFLoss_forward, global_step)

        if self.cfg["train"]["FSIMLoss_weight"] > 0:
            FSIMLoss_forward = self.cfg["train"]["FSIMLoss_weight"] * self.FSIMLoss(
                out, hr_image
            )
            total_loss += FSIMLoss_forward
            writer.add_scalar("loss/FSIM" + log_suffix, FSIMLoss_forward, global_step)

        if self.cfg["train"]["GMSDLoss_weight"] > 0:
            GMSDLoss_forward = self.cfg["train"]["GMSDLoss_weight"] * self.GMSDLoss(
                out, hr_image
            )
            total_loss += GMSDLoss_forward
            writer.add_scalar("loss/GMSD" + log_suffix, GMSDLoss_forward, global_step)

        if self.cfg["train"]["MultiScaleGMSDLoss_weight"] > 0:
            MultiScaleGMSDLoss_forward = self.cfg["train"][
                "MultiScaleGMSDLoss_weight"
            ] * self.MultiScaleGMSDLoss(out, hr_image)
            total_loss += MultiScaleGMSDLoss_forward
            writer.add_scalar(
                "loss/MultiScaleGMSD" + log_suffix,
                MultiScaleGMSDLoss_forward,
                global_step,
            )

        if self.cfg["train"]["VSILoss_weight"] > 0:
            VSILoss_forward = self.cfg["train"]["VSILoss_weight"] * (
                self.VSILoss(out, hr_image)
            )
            total_loss += VSILoss_forward
            writer.add_scalar("loss/VSI" + log_suffix, VSILoss_forward, global_step)

        if self.cfg["train"]["HaarPSILoss_weight"] > 0:
            HaarPSILoss_forward = self.cfg["train"][
                "HaarPSILoss_weight"
            ] * self.HaarPSILoss(out, hr_image)
            total_loss += HaarPSILoss_forward
            writer.add_scalar(
                "loss/HaarPSI" + log_suffix, HaarPSILoss_forward, global_step
            )

        if self.cfg["train"]["MDSILoss_weight"] > 0:
            MDSILoss_forward = self.cfg["train"]["MDSILoss_weight"] * self.MDSILoss(
                out, hr_image
            )
            total_loss += MDSILoss_forward
            writer.add_scalar("loss/DSI" + log_suffix, MDSILoss_forward, global_step)

        if self.cfg["train"]["BRISQUELoss_weight"] > 0:
            BRISQUELoss_forward = self.cfg["train"][
                "BRISQUELoss_weight"
            ] * self.BRISQUELoss(out)
            total_loss += BRISQUELoss_forward
            writer.add_scalar(
                "loss/BRISQUE" + log_suffix, BRISQUELoss_forward, global_step
            )

        if self.cfg["train"]["PieAPP_weight"] > 0:
            PieAPP_forward = self.cfg["train"]["PieAPP_weight"] * self.PieAPP(
                out, hr_image
            )
            total_loss += PieAPP_forward
            writer.add_scalar("loss/PieAPP" + log_suffix, PieAPP_forward, global_step)

        if self.cfg["train"]["DISTS_weight"] > 0:
            DISTS_forward = self.cfg["train"]["DISTS_weight"] * self.DISTS(
                out, hr_image
            )
            total_loss += DISTS_forward
            writer.add_scalar("loss/DISTS" + log_suffix, DISTS_forward, global_step)

        if self.cfg["train"]["IS_weight"] > 0:
            if self.cfg["train"]["force_piq_fp16"] is True:
                i1 = self.piq_model(out.half())
                i2 = self.piq_model(hr_image.half())
            else:
                i1 = self.piq_model(out)
                i2 = self.piq_model(hr_image)
            IS_forward = self.cfg["train"]["IS_weight"] * self.IS(i1, i2)
            total_loss += IS_forward
            writer.add_scalar("loss/IS" + log_suffix, IS_forward, global_step)

        if self.cfg["train"]["FID_weight"] > 0:
            if self.cfg["train"]["force_piq_fp16"] is True:
                i1 = self.piq_model(out.half())
                i2 = self.piq_model(hr_image.half())
            else:
                i1 = self.piq_model(out)
                i2 = self.piq_model(hr_image)
            FID_forward = self.cfg["train"]["FID_weight"] * self.FID(i1, i2)
            total_loss += FID_forward
            writer.add_scalar("loss/FID" + log_suffix, FID_forward, global_step)

        if self.cfg["train"]["KID_weight"] > 0:
            if self.cfg["train"]["force_piq_fp16"] is True:
                i1 = self.piq_model(out.half())
                i2 = self.piq_model(hr_image.half())
            else:
                i1 = self.piq_model(out)
                i2 = self.piq_model(hr_image)
            KID_forward = self.cfg["train"]["KID_weight"] * self.KID(i1, i2)
            total_loss += KID_forward
            writer.add_scalar("loss/KID" + log_suffix, KID_forward, global_step)

        if self.cfg["train"]["PR_weight"] > 0:
            precision, recall = self.PR(self.piq_model(out), self.piq_model(hr_image))
            PR_forward = self.cfg["train"]["PR_weight"] * (precision**-1)
            total_loss += PR_forward
            writer.add_scalar("loss/PR" + log_suffix, PR_forward, global_step)

        if self.cfg["train"]["Canny_weight"] > 0:
            Canny_forward = self.cfg["train"]["Canny_weight"] * self.CannyLoss(
                out, hr_image
            )
            total_loss += Canny_forward
            writer.add_scalar("loss/Canny" + log_suffix, Canny_forward, global_step)

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
            writer.add_scalar("loss/l1_stage1" + log_suffix, l1_stage1, global_step)

        # CSA Loss
        if self.cfg["network_G"]["netG"] == "CSA":
            recon_loss = self.L1Loss(other["coarse_result"], hr_image) + self.L1Loss(
                out, hr_image
            )
            cons = self.ConsistencyLoss()
            cons_loss = cons(other["csa"], other["csa_d"], hr_image, other["mask"])
            writer.add_scalar("loss/recon_loss" + log_suffix, recon_loss, global_step)
            total_loss += recon_loss
            writer.add_scalar("loss/cons_loss" + log_suffix, cons_loss, global_step)
            total_loss += cons_loss

        # EdgeConnect
        if self.cfg["network_G"]["netG"] == "EdgeConnect":
            l1_edge = self.L1Loss(other["other_img"], other["edge"])
            total_loss += l1_edge
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
            writer.add_scalar("loss/edge_big_l1" + log_suffix, edge_big_l1, global_step)
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
            writer.add_scalar("loss/mid_l1_loss" + log_suffix, mid_l1_loss, global_step)

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
            writer.add_scalar("loss/edge_loss" + log_suffix, edge_loss, global_step)

            projected_loss = (
                self.L1Loss(other["projected_image"], hr_image)
                * self.cfg["train"]["CTSDG_projected_weight"]
            )
            total_loss += projected_loss
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
            writer.add_scalar("loss/sobel_loss" + log_suffix, sobel_loss, global_step)

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
            writer.add_scalar("loss/fm_loss" + log_suffix, fm_loss, global_step)

        if self.cfg["network_D"]["netD"] is None:
            g_opt.zero_grad()
            self.manual_backward(total_loss, retain_graph=True)
            if self.cfg["train"]["gradient_clipping_G"]:
                self.clip_gradients(
                    g_opt,
                    gradient_clip_val=self.cfg["train"]["gradient_clipping_G_value"],
                    gradient_clip_algorithm="norm",
                )
            g_opt.step()

        #########################
        # discriminator
        #########################
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
                    writer.add_scalar(
                        "loss/feature_matching_loss" + log_suffix,
                        feature_matching_loss_forward,
                        global_step,
                    )

            if self.cfg["network_D"]["netD"] is not None:
                if self.cfg["datasets"]["train"]["mode"] == "DS_realesrgan":
                    hr_image = other["gt"]

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

                writer.add_scalar("loss/d_loss" + log_suffix, d_loss, global_step)

                g_opt.zero_grad()
                self.manual_backward(total_loss, retain_graph=True)
                if self.cfg["train"]["gradient_clipping_G"]:
                    self.clip_gradients(
                        g_opt,
                        gradient_clip_val=self.cfg["train"][
                            "gradient_clipping_G_value"
                        ],
                        gradient_clip_algorithm="norm",
                    )
                g_opt.step()

                d_opt.zero_grad()
                self.manual_backward(d_loss, retain_graph=True)
                if self.cfg["train"]["gradient_clipping_D"]:
                    self.clip_gradients(
                        d_opt,
                        gradient_clip_val=self.cfg["train"][
                            "gradient_clipping_D_value"
                        ],
                        gradient_clip_algorithm="norm",
                    )
                d_opt.step()

                total_loss += d_loss
        return total_loss
