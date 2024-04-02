import pytorch_lightning as pl
import torch
import yaml
from data.dataloader import DataModule
import sys


torch.set_float32_matmul_precision("medium")

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

if __name__ == "__main__":
    #############################################
    # Dataloader
    dm = DataModule(
        batch_size=cfg["datasets"]["train"]["batch_size"],
        val_lr=cfg["datasets"]["val"]["dataroot_LR"],
        val_hr=cfg["datasets"]["val"]["dataroot_HR"],
        dir_lr=cfg["datasets"]["train"]["dataroot_LR"],
        dir_hr=cfg["datasets"]["train"]["dataroot_HR"],
        num_workers=cfg["datasets"]["train"]["n_workers"],
        HR_size=cfg["datasets"]["train"]["HR_size"],
        scale=cfg["scale"],
        mask_dir=cfg["datasets"]["train"]["masks"],
        canny_min=cfg["datasets"]["train"]["canny_min"],
        canny_max=cfg["datasets"]["train"]["canny_max"],
    )
    #############################################
    # Model
    from CustomTrainClass import CustomTrainClass

    model = CustomTrainClass()
    #############################################
    # Training
    #############################################
    # GPU
    # Also maybe useful:
    # auto_cfg['scale']_batch_size='binsearch'
    # stochastic_weight_avg=True

    # disable validation
    # limit_val_batches=0

    # Warning: stochastic_weight_avg **can cause crashing after an epoch**. Test if it crashes first if you reach next epoch. Not all generators are tested.
    if cfg["use_tpu"] is False:
        trainer = pl.Trainer(
            num_sanity_val_steps=0,
            log_every_n_steps=50,
            check_val_every_n_epoch=None,
            val_check_interval=int(cfg["datasets"]["train"]["save_step_frequency"]),
            logger=None,
            accelerator="gpu",
            devices=cfg["gpus"],
            precision=32,
            max_epochs=cfg["datasets"]["train"]["max_epochs"],
            default_root_dir=cfg["default_root_dir"],
        )

    # 2+ cfg['gpus'] (locally, not inside Google Colab)
    # Recommended: Pytorch 1.8+. 1.7.1 seems to have dataloader issues and ddp only works if code is run within console.
    if cfg["use_tpu"] is False and cfg["gpus"] > 1:
        trainer = pl.Trainer(
            num_sanity_val_steps=0,
            log_every_n_steps=50,
            resume_from_checkpoint=cfg["path"]["checkpoint_path"],
            check_val_every_n_epoch=None,
            val_check_interval=int(cfg["datasets"]["train"]["save_step_frequency"]),
            logger=None,
            accelerator="gpu",
            devices=cfg["gpus"],
            precision=32,
            strategy=cfg["distributed_backend"],
            max_epochs=cfg["datasets"]["train"]["max_epochs"],
            default_root_dir=cfg["default_root_dir"],
        )

    # TPU
    if cfg["use_tpu"] is True and cfg["use_amp"] is False:
        print("Currently not supported")
        sys.exit(0)

    if cfg["use_tpu"] is True and cfg["use_amp"] is True:
        print("Currently not supported")
        sys.exit(0)

    # Loading a pretrain pth
    if cfg["path"]["pretrain_model_G"]:

        # model.netG.load_state_dict(torch.load(cfg['path']['pretrain_model_G'])['state_dict'])
        model.netG.load_state_dict(
            torch.load(cfg["path"]["pretrain_model_G"]), strict=False
        )
        print("Pretrain Generator pth loaded!")

    if cfg["path"]["pretrain_model_D"]:
        model.netD.load_state_dict(torch.load(cfg["path"]["pretrain_model_D"]))
        print("Pretrain Discriminator pth loaded!")

    if cfg["path"]["pretrain_model_G_teacher"]:
        model.netG_teacher.load_state_dict(
            torch.load(cfg["path"]["pretrain_model_G_teacher"]), strict=True
        )
        print("Teacher pth loaded!")

    #############################################

    #############################################
    # Loading a Model
    #############################################
    # For resuming training
    if cfg["path"]["checkpoint_path"] is not None:
        # load from checkpoint (optional) (using a model as pretrain and disregarding other parameters)
        # model = model.load_from_checkpoint(checkpoint_path) # start training from checkpoint, warning: apperantly global_step will be reset to zero and overwriting validation images, you could manually make an offset
        # model = model.load_from_checkpoint(cfg['path']['checkpoint_path'])

        # continue training with checkpoint (does restore values) (optional)
        # https://github.com/PyTorchLightning/pytorch-lightning/issues613
        # https://pytorch-lightning.readthedocs.io/en/0.6.0/pytorch_lightning.trainer.training_io.html
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4333
        # dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'callbacks', 'optimizer_states', 'lr_schedulers', 'state_dict', 'hparams_name', 'hyper_parameters'])

        # To use DDP for local multi-GPU training, you need to add find_unused_parameters=True inside the DDP command
        model = model.load_from_checkpoint(cfg["path"]["checkpoint_path"], strict=False)
        # trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, logger=None, gpus=cfg['gpus'], max_epochs=cfg['datasets']['train']['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'], default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['datasets']['train']['save_step_frequency'], save_path = cfg['path']['checkpoint_save_path'])])
        checkpoint = torch.load(cfg["path"]["checkpoint_path"])
        # trainer.checkpoint_connector.restore(checkpoint)
        # trainer.checkpoint_connector.restore_training_state(checkpoint)
        pl.Trainer.global_step = checkpoint["global_step"]
        pl.Trainer.epoch = checkpoint["epoch"]

        print("Checkpoint was loaded successfully.")

    #############################################

    if cfg["path"]["checkpoint_path"]:
        trainer.strategy.strict_loading = False
        # trainer.fit(model, dm, ckpt_path=cfg["path"]["checkpoint_path"])
        trainer.fit(model, dm)
    else:
        trainer.fit(model, dm)
