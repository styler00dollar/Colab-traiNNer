import pytorch_lightning as pl
#%cd /content/

dir_lr = '/content/lr/' #@param
dir_hr = '/content/hr/' #@param
val_lr =  '/content/val_lr/'#@param
val_hr = '/content/val_hr/' #@param
num_workers = 1 #@param
hr_size = 256 #@param
scale = 4 #@param
batch_size = 1 #@param
batch_size_DL = 20 #@param
gpus=1 #@param
max_epochs = 100 #@param
progress_bar_refresh_rate = 20 #@param
default_root_dir='/content/' #@param
save_path='/content/' #@param
save_step_frequency = 100 #@param
tpu_cores = 8 #@param
#@markdown For batch dataloader
image_size=400 #@param
amount_tiles=3 #@param
#############################################
# Dataloader
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
model = CustomTrainClass()

#@markdown Loading a pretrain pth
pretrain_path = None #@param
if pretrain_path is not None:
  trainer.model.netG.load_state_dict(torch.load(model_path))

#@markdown For resuming training
checkpoint_path = '/content/Checkpoint_1_500.ckpt' #@param

# load from checkpoint (optional) (using a model as pretrain and disregarding other parameters)
#model = model.load_from_checkpoint(checkpoint_path) # start training from checkpoint, warning: apperantly global_step will be reset to zero and overwriting validation images, you could manually make an offset


# continue training with checkpoint (does restore values) (optional)
# https://github.com/PyTorchLightning/pytorch-lightning/issues/2613
# https://pytorch-lightning.readthedocs.io/en/0.6.0/pytorch_lightning.trainer.training_io.html
# https://github.com/PyTorchLightning/pytorch-lightning/issues/4333
# dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'callbacks', 'optimizer_states', 'lr_schedulers', 'state_dict', 'hparams_name', 'hyper_parameters'])

# To use DDP for local multi-GPU training, you need to add find_unused_parameters=True inside the DDP command
"""
model = model.load_from_checkpoint(checkpoint_path)
trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, logger=None, gpus=1, max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=100, save_path='/content/')])
checkpoint = torch.load(checkpoint_path)
trainer.checkpoint_connector.restore(checkpoint, on_gpu=True)
trainer.checkpoint_connector.restore_training_state(checkpoint)
pl.Trainer.global_step = checkpoint['global_step']
pl.Trainer.epoch = checkpoint['epoch']
"""
#############################################



#############################################
# Training
#############################################
# GPU
# Also maybe useful:
# auto_scale_batch_size='binsearch'
# stochastic_weight_avg=True

# Warning: stochastic_weight_avg **can cause crashing after an epoch**. Test if it crashes first if you reach next epoch. Not all generators are tested.
trainer = pl.Trainer(logger=None, gpus=gpus, max_epochs=max_epochs, progress_bar_refresh_rate=progress_bar_refresh_rate, default_root_dir=default_root_dir, callbacks=[CheckpointEveryNSteps(save_step_frequency=save_step_frequency, save_path=save_path)])
# 2+ GPUS (locally, not inside Google Colab)
# Recommended: Pytorch 1.8+. 1.7.1 seems to have dataloader issues and ddp only works if code is run within console.
#trainer = pl.Trainer(logger=None, gpus=2, distributed_backend='dp', max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=100, save_path='/content/')])
# GPU with AMP (amp_level='O1' = mixed precision, 'O2' = Almost FP16, 'O3' = FP16)
# https://nvidia.github.io/apex/amp.html?highlight=opt_level#o1-mixed-precision-recommended-for-typical-use
#trainer = pl.Trainer(logger=None, gpus=1, precision=16, amp_level='O1', max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=1000, save_path='/content/')])
# TPU
#trainer = pl.Trainer(logger=None, tpu_cores=tpu_cores, max_epochs=max_epochs, progress_bar_refresh_rate=progress_bar_refresh_rate, default_root_dir=default_root_dir, callbacks=[CheckpointEveryNSteps(save_step_frequency=save_step_frequency, save_path=save_path)])
#############################################

trainer.fit(model, dm)
