import pytorch_lightning as pl
#%cd /content/

import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


dir_lr = cfg['datasets']['train']['dataroot_LR']
dir_hr = cfg['datasets']['train']['dataroot_HR']
val_lr =  cfg['datasets']['val']['dataroot_LR']
val_hr = cfg['datasets']['val']['dataroot_HR']
num_workers = cfg['datasets']['train']['n_workers']
hr_size = cfg['datasets']['train']['HR_size']
scale = cfg['scale']
batch_size = cfg['datasets']['train']['batch_size']
batch_size_DL = cfg['datasets']['train']['batch_size_DL']
gpus=cfg['gpus']
max_epochs = cfg['datasets']['train']['max_epochs']
progress_bar_refresh_rate = cfg['progress_bar_refresh_rate']
default_root_dir=cfg['default_root_dir']
save_path=cfg['path']['checkpoint_save_path']
save_step_frequency = cfg['datasets']['train']['save_step_frequency']
tpu_cores = cfg['tpu_cores']
# batch dataloader
image_size=cfg['datasets']['train']['image_size']
amount_tiles=cfg['datasets']['train']['amount_tiles']
#############################################
# Dataloader
from data.dataloader import DFNetDataModule
#############################################
# Inpainting
# normal training
#dm = DFNetDataModule(batch_size=batch_size, training_path = dir_hr, validation_path = val_lr)
# tiled dataloader (batch return)
#dm = DFNetDataModule(training_path = dir_hr, validation_path = val_lr, batch_size=batch_size, num_workers=num_workers, batch_size_DL=batch_size_DL)

# Super Resolution
# lr/hr dataloader
dm = DFNetDataModule(batch_size=batch_size, dir_lr = dir_lr, dir_hr = dir_hr, val_lr = val_lr, val_hr = val_hr, num_workers = num_workers, hr_size = hr_size, scale = scale)
# batch
#dm = DFNetDataModule(batch_size=batch_size, training_path = dir_lr, val_lr = val_lr, val_hr = val_hr, num_workers=num_workers, batch_size_DL=batch_size_DL, hr_size=hr_size, scale = scale, image_size=image_size, amount_tiles=amount_tiles)
#############################################


#############################################
# Loading a Model
#############################################
from CustomTrainClass import CustomTrainClass
model = CustomTrainClass()

# For resuming training
if cfg['path']['checkpoint_path'] is not None:
  checkpoint_path = cfg['path']['checkpoint_path']

  # load from checkpoint (optional) (using a model as pretrain and disregarding other parameters)
  #model = model.load_from_checkpoint(checkpoint_path) # start training from checkpoint, warning: apperantly global_step will be reset to zero and overwriting validation images, you could manually make an offset



  # continue training with checkpoint (does restore values) (optional)
  # https://github.com/PyTorchLightning/pytorch-lightning/issues/2613
  # https://pytorch-lightning.readthedocs.io/en/0.6.0/pytorch_lightning.trainer.training_io.html
  # https://github.com/PyTorchLightning/pytorch-lightning/issues/4333
  # dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'callbacks', 'optimizer_states', 'lr_schedulers', 'state_dict', 'hparams_name', 'hyper_parameters'])

  # To use DDP for local multi-GPU training, you need to add find_unused_parameters=True inside the DDP command

  model = model.load_from_checkpoint(checkpoint_path)
  trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, logger=None, gpus=1, max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=100, save_path='/content/')])
  checkpoint = torch.load(checkpoint_path)
  trainer.checkpoint_connector.restore(checkpoint, on_gpu=True)
  trainer.checkpoint_connector.restore_training_state(checkpoint)
  pl.Trainer.global_step = checkpoint['global_step']
  pl.Trainer.epoch = checkpoint['epoch']

#############################################



#############################################
# Training
#############################################
# GPU
# Also maybe useful:
# auto_scale_batch_size='binsearch'
# stochastic_weight_avg=True
from checkpoint import CheckpointEveryNSteps

# Warning: stochastic_weight_avg **can cause crashing after an epoch**. Test if it crashes first if you reach next epoch. Not all generators are tested.
if cfg['use_tpu'] == False and cfg['use_amp'] == False:
  trainer = pl.Trainer(logger=None, gpus=gpus, max_epochs=max_epochs, progress_bar_refresh_rate=progress_bar_refresh_rate, default_root_dir=default_root_dir, callbacks=[CheckpointEveryNSteps(save_step_frequency=save_step_frequency, save_path=save_path)])
# GPU with AMP (amp_level='O1' = mixed precision, 'O2' = Almost FP16, 'O3' = FP16)
# https://nvidia.github.io/apex/amp.html?highlight=opt_level#o1-mixed-precision-recommended-for-typical-use
if cfg['use_tpu'] == False and cfg['use_amp'] == True:
  trainer = pl.Trainer(logger=None, gpus=1, precision=16, amp_level='O1', max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=1000, save_path='/content/')])

# 2+ GPUS (locally, not inside Google Colab)
# Recommended: Pytorch 1.8+. 1.7.1 seems to have dataloader issues and ddp only works if code is run within console.
if cfg['use_tpu'] == False and cfg['gpus'] > 1 and cfg['use_amp'] == False:
  trainer = pl.Trainer(logger=None, gpus=cfg['gpus'], distributed_backend='dp', max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=100, save_path='/content/')])

if cfg['use_tpu'] == False and cfg['gpus'] > 1 and cfg['use_amp'] == True:
  trainer = pl.Trainer(logger=None, gpus=cfg['gpus'], precision=16, distributed_backend='dp', max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=100, save_path='/content/')])

# TPU
if cfg['use_tpu'] == True and cfg['use_amp'] == False:
  trainer = pl.Trainer(logger=None, tpu_cores=cfg['tpu_cores'], max_epochs=max_epochs, progress_bar_refresh_rate=progress_bar_refresh_rate, default_root_dir=default_root_dir, callbacks=[CheckpointEveryNSteps(save_step_frequency=save_step_frequency, save_path=save_path)])

if cfg['use_tpu'] == True and cfg['use_amp'] == True:
  trainer = pl.Trainer(logger=None, tpu_cores=cfg['tpu_cores'], precision=16, max_epochs=max_epochs, progress_bar_refresh_rate=progress_bar_refresh_rate, default_root_dir=default_root_dir, callbacks=[CheckpointEveryNSteps(save_step_frequency=save_step_frequency, save_path=save_path)])



# Loading a pretrain pth
if cfg['path']['pretrain_model_G']:
  trainer.model.netG.load_state_dict(torch.load(cfg['path']['pretrain_model_G']))

#############################################

trainer.fit(model, dm)
