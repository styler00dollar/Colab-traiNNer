#https://github.com/PyTorchLightning/pytorch-lightning/issues/2534
import os
import pytorch_lightning as pl
import torch

import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="Checkpoint",
        use_modelcheckpoint_filename=False,
        save_path = '/content/'
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.save_path = save_path

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            #ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            ckpt_path = os.path.join(cfg['path']['checkpoint_save_path'], filename)
            trainer.save_checkpoint(ckpt_path)

            # saving normal .pth models
            #https://github.com/PyTorchLightning/pytorch-lightning/issues/4114
            torch.save(trainer.model.netG.state_dict(), os.path.join(cfg['path']['checkpoint_save_path'], f"{self.prefix}_{epoch}_{global_step}_G.pth"))
            if cfg['network_D']['netD'] != None:
              torch.save(trainer.model.netD.state_dict(), os.path.join(cfg['path']['checkpoint_save_path'], f"{self.prefix}_{epoch}_{global_step}_D.pth"))

            # saving jit for cain
            if cfg['network_G']['netG'] == 'CAIN':
                traced_model = torch.jit.trace(trainer.model.netG, (torch.randn(1,3,256,256).cuda(),torch.randn(1,3,256,256).cuda()))
                torch.jit.save(traced_model, os.path.join(cfg['path']['checkpoint_save_path'], f"{self.prefix}_{epoch}_{global_step}_G.pt"))
            
            # run validation once checkpoint was made
            trainer.run_evaluation()

    #def on_epoch_end(self, trainer: pl.Trainer, _):
    #    print("Epoch completed.")

    def on_train_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        ckpt_path = os.path.join(self.save_path, f"{self.prefix}_{epoch}_{global_step}.ckpt")
        trainer.save_checkpoint(ckpt_path)
        print("Checkpoint " + f"{self.prefix}_{epoch}_{global_step}.ckpt" + " saved.")

        torch.save(trainer.model.netG.state_dict(), os.path.join(cfg['path']['checkpoint_save_path'], f"{self.prefix}_{epoch}_{global_step}_G.pth"))
        if cfg['network_D']['netD'] != None:
          torch.save(trainer.model.netD.state_dict(), os.path.join(cfg['path']['checkpoint_save_path'], f"{self.prefix}_{epoch}_{global_step}_D.pth"))
          print("Checkpoint " + f"{self.prefix}_{epoch}_{global_step}_G.pth" + " and " + f"{self.prefix}_{epoch}_{global_step}_D.pth" + "saved")
        else:
          print("Checkpoint " + f"{self.prefix}_{epoch}_{global_step}_G.pth saved")

        if cfg['network_G']['netG'] == 'CAIN':
            traced_model = torch.jit.trace(trainer.model.netG, (torch.randn(1,3,256,256).cuda(),torch.randn(1,3,256,256).cuda()))
            torch.jit.save(traced_model, os.path.join(cfg['path']['checkpoint_save_path'], f"{self.prefix}_{epoch}_{global_step}_G.pt"))
        
#Trainer(callbacks=[CheckpointEveryNSteps()])
