from data.data import DS_inpaint_val
from CustomTrainClass import CustomTrainClass
import pytorch_lightning as pl

# Warning: make sure config.yaml does reflect your settings during training
data_input_folder = '/content/val_lr/' # images that are masked with green
netG_pth_path = '/content/drive/MyDrive/Colab-BasicSR/lightning/Checkpoint_39_100000_G.pth'
# output path currently defined in config file

# init
dm = DS_inpaint_val(data_input_folder)
model = CustomTrainClass()

# load pth generator
import torch
model.netG.load_state_dict(torch.load(netG_pth_path))

# GPU
trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=20)
# GPU with AMP (amp_level='O1' = mixed precision)
#trainer = pl.Trainer(gpus=1, precision=16, amp_level='O1', progress_bar_refresh_rate=20)
# TPU
#trainer = pl.Trainer(tpu_cores=8, progress_bar_refresh_rate=20)

trainer.test(model, dm)
