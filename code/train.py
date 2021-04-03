import pytorch_lightning as pl
import torch
import yaml
from checkpoint import CheckpointEveryNSteps
from data.dataloader import DFNetDataModule

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

if __name__ == '__main__':
  #############################################
  # Dataloader
  dm = DFNetDataModule(batch_size=cfg['datasets']['train']['batch_size'], val_lr = cfg['datasets']['val']['dataroot_LR'], val_hr = cfg['datasets']['val']['dataroot_HR'], dir_lr = cfg['datasets']['train']['dataroot_LR'], dir_hr = cfg['datasets']['train']['dataroot_HR'], num_workers = cfg['datasets']['train']['n_workers'], HR_size = cfg['datasets']['train']['HR_size'], scale = cfg['scale'], mask_dir=cfg['datasets']['train']['masks'], batch_size_DL=cfg['datasets']['train']['batch_size_DL'], image_size=cfg['datasets']['train']['image_size'], amount_tiles = cfg['datasets']['train']['amount_tiles'], canny_min = cfg['datasets']['train']['canny_min'], canny_max = cfg['datasets']['train']['canny_max'])
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
  if cfg['use_tpu'] == False and cfg['use_amp'] == False:
    trainer = pl.Trainer(log_every_n_steps=50, resume_from_checkpoint=cfg['path']['checkpoint_path'], check_val_every_n_epoch=9999999, logger=None, gpus=cfg['gpus'], max_epochs=cfg['datasets']['train']['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'], default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['datasets']['train']['save_step_frequency'], save_path=cfg['path']['checkpoint_save_path'])])
  # GPU with AMP (amp_level='O1' = mixed precision, 'O2' = Almost FP16, 'O3' = FP16)
  # https://nvidia.github.io/apex/amp.html?highlight=opt_level#o1-mixed-precision-recommended-for-typical-use
  if cfg['use_tpu'] == False and cfg['use_amp'] == True:
    trainer = pl.Trainer(log_every_n_steps=50, resume_from_checkpoint=cfg['path']['checkpoint_path'], check_val_every_n_epoch=9999999, logger=None, gpus=cfg['gpus'], precision=16, amp_level='O1', max_epochs=cfg['datasets']['train']['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'], default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['datasets']['train']['save_step_frequency'], save_path=cfg['path']['checkpoint_save_path'])])

  # 2+ cfg['gpus'] (locally, not inside Google Colab)
  # Recommended: Pytorch 1.8+. 1.7.1 seems to have dataloader issues and ddp only works if code is run within console.
  if cfg['use_tpu'] == False and cfg['gpus'] > 1 and cfg['use_amp'] == False:
    trainer = pl.Trainer(log_every_n_steps=50, resume_from_checkpoint=cfg['path']['checkpoint_path'], check_val_every_n_epoch=9999999, logger=None, gpus=cfg['gpus'], distributed_backend='dp', max_epochs=cfg['datasets']['train']['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'], default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['datasets']['train']['save_step_frequency'], checkpsave_pathoint_save_path=cfg['path']['checkpoint_save_path'])])

  if cfg['use_tpu'] == False and cfg['gpus'] > 1 and cfg['use_amp'] == True:
    trainer = pl.Trainer(log_every_n_steps=50, resume_from_checkpoint=cfg['path']['checkpoint_path'], check_val_every_n_epoch=9999999, logger=None, gpus=cfg['gpus'], precision=16, distributed_backend='dp', max_epochs=cfg['datasets']['train']['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'],default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['datasets']['train']['save_step_frequency'], save_path=cfg['path']['checkpoint_save_path'])])

  # TPU
  if cfg['use_tpu'] == True and cfg['use_amp'] == False:
    trainer = pl.Trainer(log_every_n_steps=50, resume_from_checkpoint=cfg['path']['checkpoint_path'], check_val_every_n_epoch=9999999, logger=None, tpu_cores=cfg['tpu_cores'],max_epochs=cfg['datasets']['train']['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'], default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['datasets']['train']['save_step_frequency'], save_path=cfg['path']['checkpoint_save_path'])])

  if cfg['use_tpu'] == True and cfg['use_amp'] == True:
    trainer = pl.Trainer(log_every_n_steps=50, resume_from_checkpoint=cfg['path']['checkpoint_path'], check_val_every_n_epoch=9999999, logger=None, tpu_cores=cfg['tpu_cores'], precision=16, max_epochs=cfg['datasets']['train']['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'], default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['datasets']['train']['save_step_frequency'], save_path=cfg['path']['checkpoint_save_path'])])

  # Loading a pretrain pth
  if cfg['path']['pretrain_model_G']:
    model.netG.load_state_dict(torch.load(cfg['path']['pretrain_model_G']))

  #############################################



  #############################################
  # Loading a Model
  #############################################
  # For resuming training
  if cfg['path']['checkpoint_path'] is not None:
    # load from checkpoint (optional) (using a model as pretrain and disregarding other parameters)
    #model = model.load_from_checkpoint(checkpoint_path) # start training from checkpoint, warning: apperantly global_step will be reset to zero and overwriting validation images, you could manually make an offset

    # continue training with checkpoint (does restore values) (optional)
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2613
    # https://pytorch-lightning.readthedocs.io/en/0.6.0/pytorch_lightning.trainer.training_io.html
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/4333
    # dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'callbacks', 'optimizer_states', 'lr_schedulers', 'state_dict', 'hparams_name', 'hyper_parameters'])

    # To use DDP for local multi-GPU training, you need to add find_unused_parameters=True inside the DDP command
    model = model.load_from_checkpoint(cfg['path']['checkpoint_path'])
    #trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, logger=None, gpus=cfg['gpus'], max_epochs=cfg['datasets']['train']['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'], default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['datasets']['train']['save_step_frequency'], save_path = cfg['path']['checkpoint_save_path'])])
    checkpoint = torch.load(cfg['path']['checkpoint_path'])
    trainer.checkpoint_connector.restore(checkpoint, on_gpu=True)
    trainer.checkpoint_connector.restore_training_state(checkpoint)
    pl.Trainer.global_step = checkpoint['global_step']
    pl.Trainer.epoch = checkpoint['epoch']
    print("Checkpoint was loaded successfully.")

  #############################################


  trainer.fit(model, dm)
