# https://github.com/PyTorchLightning/pytorch-lightning/issues/2534
import os
import pytorch_lightning as pl
import torch

import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


class CheckpointOnInterrupt(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        prefix="Checkpoint",
        save_path="/content/",
    ):
        self.prefix = prefix
        self.save_path = save_path

    def on_keyboard_interrupt(self, trainer, pl_module):
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        ckpt_path = os.path.join(
            self.save_path, f"{self.prefix}_{epoch}_{global_step}.ckpt"
        )
        trainer.save_checkpoint(ckpt_path)
        print("Checkpoint " + f"{self.prefix}_{epoch}_{global_step}.ckpt" + " saved.")

        torch.save(
            trainer.model.netG.state_dict(),
            os.path.join(
                cfg["path"]["checkpoint_save_path"],
                f"{self.prefix}_{epoch}_{global_step}_G.pth",
            ),
        )
        if cfg["network_D"]["netD"] != None:
            torch.save(
                trainer.model.netD.state_dict(),
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
                trainer.model.netG,
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
