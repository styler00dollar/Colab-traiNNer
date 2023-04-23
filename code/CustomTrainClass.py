import yaml
import cv2
from loss.metrics import *
from torchvision.utils import save_image
import pytorch_lightning as pl
from init import weights_init
import os
import numpy as np
from tensorboardX import SummaryWriter
from generate import generate
from check_arch import check_arch

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

writer = SummaryWriter(logdir=cfg["path"]["log_path"])


class CustomTrainClass(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

        ##################################################################

        from generator import CreateGenerator

        self.netG = CreateGenerator(cfg["network_G"], cfg["scale"])

        if cfg["network_G_teacher"]["netG"] != None:
            print("Using Teacher!")
            self.netG_teacher = CreateGenerator(cfg["network_G_teacher"], cfg["scale"])
            for param in self.netG_teacher.parameters():
                param.requires_grad = False

        if (
            cfg["path"]["checkpoint_path"] is None
            and cfg["network_G"]["netG"] != "GLEAN"
            and cfg["network_G"]["netG"] != "srflow"
            and cfg["network_G"]["netG"] != "GFPGAN"
            and cfg["network_G"]["netG"] != "GMFSS_union"
        ):
            if self.global_step == 0:
                weights_init(self.netG, "kaiming")
                print("Generator weight init complete.")

        ##################################################################

        if cfg["network_D"]["netD"] != None:
            from discriminator import CreateDiscriminator

            self.netD = CreateDiscriminator(cfg)

            # only doing init, if not 'TranformerDiscriminator', 'EfficientNet',
            # 'ResNeSt', 'resnet', 'ViT', 'DeepViT', 'mobilenetV3'
            if cfg["network_D"]["netD"] in (
                "resnet3d",
                "NFNet",
                "context_encoder",
                "VGG",
                "VGG_fea",
                "Discriminator_VGG_128_SN",
                "VGGFeatureExtractor",
                "NLayerDiscriminator",
                "MultiscaleDiscriminator",
                "Discriminator_ResNet_128",
                "ResNet101FeatureExtractor",
                "MINCNet",
                "PixelDiscriminator",
                "ResNeSt",
                "RepVGG",
                "squeezenet",
                "SwinTransformer",
            ):
                if self.global_step == 0:
                    weights_init(self.netD, "kaiming")
                    print("Discriminator weight init complete.")
        else:
            self.netD = None  # Passing none into loss calc

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
        if "PSNR" in cfg["train"]["metrics"]:
            self.val_psnr = []
        if "SSIM" in cfg["train"]["metrics"]:
            self.val_ssim = []
        if "MSE" in cfg["train"]["metrics"]:
            self.val_mse = []
        if "LPIPS" in cfg["train"]["metrics"]:
            self.val_lpips = []

        self.iter_check = 0

        if (
            cfg["train"]["KID_weight"] > 0
            or cfg["train"]["IS_weight"] > 0
            or cfg["train"]["FID_weight"] > 0
            or cfg["train"]["PR_weight"] > 0
        ):
            from loss.inceptionV3 import fid_inception_v3

            self.piq_model = fid_inception_v3()
            self.piq_model = self.piq_model.cuda().eval()
            if cfg["train"]["force_piq_fp16"] is True:
                self.piq_model = self.piq_model.half()

        if cfg["datasets"]["train"]["mode"] == "DS_realesrgan":
            from data.realesrgan import RealESRGANDatasetApply

            self.RealESRGANDatasetApply = RealESRGANDatasetApply(self.device)

    def forward(self, image, masks):
        return self.netG(image, masks)

    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        # iteration count is sometimes broken, adding a check and manual increment
        # only increment if generator gets trained (loop gets called a second time for discriminator)
        if cfg["path"]["checkpoint_path"] is not None:
            if self.iter_check == self.trainer.global_step:
                self.trainer.global_step += 1
            self.iter_check = self.trainer.global_step

        # different networks require different data and have different data loaders
        # due to overlap, a second check for dataloader mode is needed

        # if more than one output, fills dict with data, otherwise give empty dict to loss calc
        other = dict()

        arch, edge, grayscale, landmarks = check_arch(cfg)

        # inpainting
        if arch == "inpainting" and edge:
            other["edge"] = train_batch[3]
        if arch == "inpainting" and grayscale:
            other["grayscale"] = train_batch[4]
        if arch == "inpainting":
            lr_image = train_batch[0]
            hr_image = train_batch[2]
            other["mask"] = train_batch[1]

        # interpolation
        elif arch == "interpolation":
            other["hr_image1"] = train_batch[0]
            other["hr_image3"] = train_batch[1]
            hr_image = train_batch[2]

        # sr
        if arch == "sr" and cfg["datasets"]["train"]["mode"] == "DS_realesrgan":
            lr_image, hr_image, other["gt"] = self.RealESRGANDatasetApply.forward(
                train_batch[0],
                train_batch[1],
                train_batch[2],
                train_batch[3],
                self.device,
            )
            # hotfix: at the end of one epoch it can happen that only 3d tensor gets returned
            if lr_image.dim() == 3:
                lr_image = lr_image.unsqueeze(0)
                hr_image = hr_image.unsqueeze(0)
                other["gt"] = other["gt"].unsqueeze(0)
        else:
            lr_image = train_batch[1]
            hr_image = train_batch[2]
        if arch == "sr" and landmarks:
            other["landmarks"] = train_batch[3]

        if cfg["network_G_teacher"]["netG"] != None:
            # creating dict for teacher, currently only using same lr data
            other_teacher = other.copy()

        total_loss = 0

        out, other = generate(
            cfg=cfg,
            lr_image=lr_image,
            hr_image=hr_image,
            netG=self.netG,
            other=other,
            global_step=self.trainer.global_step,
            arch=arch,
            arch_name=cfg["network_G"]["netG"],
        )

        total_loss += self.loss(
            out=out,
            hr_image=hr_image,
            writer=writer,
            global_step=self.trainer.global_step,
            optimizer_idx=optimizer_idx,
            netD=self.netD,
            other=other,
        )

        if cfg["network_G_teacher"]["netG"] != None:
            out_teacher, other_teacher = generate(
                cfg=cfg,
                lr_image=lr_image,
                hr_image=hr_image,
                netG=self.netG_teacher,
                other=other_teacher,
                global_step=self.trainer.global_step,
                arch=arch,
                arch_name=cfg["network_G_teacher"]["netG"],
            )

            total_loss += self.loss(
                out=out,
                hr_image=out_teacher,
                writer=writer,
                global_step=self.trainer.global_step,
                optimizer_idx=optimizer_idx,
                netD=self.netD,
                other=other,
                other_teacher=other_teacher,
                log_suffix="_teacher",
            )

        return total_loss

    def configure_optimizers(self):
        if cfg["network_G"]["finetune"] is True:
            input_G = self.netG.parameters()
        else:
            input_G = filter(lambda p: p.requires_grad, self.netG.parameters())

        from optimizer import CreateOptimizer

        if cfg["network_D"]["netD"] is not None:
            input_D = self.netD.parameters()
            opt_g, opt_d = CreateOptimizer(cfg, input_G, input_D)
            return [opt_g, opt_d], []
        else:
            opt_g, _ = CreateOptimizer(cfg, input_G)
            return [opt_g], []

    def validation_step(self, train_batch, train_idx):
        arch, edge, grayscale, landmarks = check_arch(cfg)
        other = dict()

        # inpainting
        if arch == "inpainting" and edge:
            other["edge"] = train_batch[3]
        if arch == "inpainting" and grayscale:
            other["grayscale"] = train_batch[4]
        if arch == "inpainting":
            lr_image = train_batch[0]
            other["mask"] = train_batch[1]

        # interpolation
        elif arch == "interpolation":
            other["hr_image1"], other["hr_image3"] = train_batch[0]
            lr_image = None
            hr_image = None

        # sr
        elif arch == "sr":
            lr_image = train_batch[0]
            hr_image = train_batch[1]
            if arch == "sr" and landmarks:
                other["landmarks"] = train_batch[3]

        path = train_batch[2]

        #########################

        out, _ = generate(
            cfg,
            lr_image,
            hr_image,
            self.netG,
            other,
            self.trainer.global_step,
            arch,
            cfg["network_G"]["netG"],
        )

        # Validation metrics work, but they need an origial source image.
        if "PSNR" in cfg["train"]["metrics"]:
            self.val_psnr.append(self.psnr_metric(hr_image, out).item())
        if "SSIM" in cfg["train"]["metrics"]:
            self.val_ssim.append(self.ssim_metric(hr_image, out).item())
        if "MSE" in cfg["train"]["metrics"]:
            self.val_mse.append(self.mse_metric(hr_image, out).item())
        if "LPIPS" in cfg["train"]["metrics"]:
            self.val_lpips.append(self.PerceptualLoss(out, hr_image).item())

        validation_output = cfg["path"]["validation_output_path"]

        # path can contain multiple files, depending on the batch_size
        for f in path:
            # data is processed as a batch, to save indididual files, a counter is used
            counter = 0
            if not os.path.exists(
                os.path.join(
                    validation_output, os.path.splitext(os.path.basename(f))[0]
                )
            ):
                os.makedirs(
                    os.path.join(
                        validation_output, os.path.splitext(os.path.basename(f))[0]
                    )
                )

            filename_with_extention = os.path.basename(f)
            filename = os.path.splitext(filename_with_extention)[0]

            # currently only supports batch_size 1
            if arch == "interpolation":
                out = out.data.mul(255).mul(255 / 255).clamp(0, 255).round()
                out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()  # *255
                out = out.astype(np.uint8)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(
                        validation_output,
                        filename,
                        str(self.trainer.global_step) + ".png",
                    ),
                    out,
                )
            else:
                save_image(
                    out[counter],
                    os.path.join(
                        validation_output,
                        filename,
                        str(self.trainer.global_step) + ".png",
                    ),
                )

            counter += 1

    def on_validation_epoch_end(self):
        self.save_checkpoint()

        if "PSNR" in cfg["train"]["metrics"]:
            val_psnr = np.mean(self.val_psnr)
            writer.add_scalar("metrics/PSNR", val_psnr, self.trainer.global_step)
            self.val_psnr = []
        if "SSIM" in cfg["train"]["metrics"]:
            val_ssim = np.mean(self.val_ssim)
            writer.add_scalar("metrics/SSIM", val_ssim, self.trainer.global_step)
            self.val_ssim = []
        if "MSE" in cfg["train"]["metrics"]:
            val_mse = np.mean(self.val_mse)
            writer.add_scalar("metrics/MSE", val_mse, self.trainer.global_step)
            self.val_mse = []
        if "LPIPS" in cfg["train"]["metrics"]:
            val_lpips = np.mean(self.val_lpips)
            writer.add_scalar("metrics/LPIPS", val_lpips, self.trainer.global_step)
            self.val_lpips = []

    def save_checkpoint(self):
        # todo: read from config
        self.prefix = "Checkpoint"

        epoch = self.trainer.current_epoch
        global_step = self.trainer.global_step
        ckpt_path = os.path.join(
            cfg["path"]["checkpoint_save_path"],
            f"{self.prefix}_{epoch}_{global_step}.ckpt",
        )
        self.trainer.save_checkpoint(ckpt_path)
        print("Checkpoint " + f"{self.prefix}_{epoch}_{global_step}.ckpt" + " saved.")

        torch.save(
            self.trainer.model.netG.state_dict(),
            os.path.join(
                cfg["path"]["checkpoint_save_path"],
                f"{self.prefix}_{epoch}_{global_step}_G.pth",
            ),
        )
        if cfg["network_D"]["netD"] != None:
            torch.save(
                self.trainer.model.netD.state_dict(),
                os.path.join(
                    cfg["path"]["checkpoint_save_path"],
                    f"{self.prefix}_{epoch}_{global_step}_D.pth",
                ),
            )
            print(
                "Checkpoint "
                + f"{self.prefix}_{epoch}_{global_step}_G.pth"
                + " and "
                + f"{self.prefix}_{epoch}_{global_step}_D.pth"
                + " saved"
            )
        else:
            print("Checkpoint " + f"{self.prefix}_{epoch}_{global_step}_G.pth saved")

        if cfg["network_G"]["netG"] == "CAIN":
            traced_model = torch.jit.trace(
                self.trainer.model.netG,
                (
                    torch.randn(1, 3, 256, 256).cuda(),
                    torch.randn(1, 3, 256, 256).cuda(),
                ),
            )
            torch.jit.save(
                traced_model,
                os.path.join(
                    cfg["path"]["checkpoint_save_path"],
                    f"{self.prefix}_{epoch}_{global_step}_G.pt",
                ),
            )
