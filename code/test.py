from data.data import DS_inpaint_val
from CustomTrainClass import CustomTrainClass
import pytorch_lightning as pl
import torch
import argparse

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_input_folder', type=str, required=True, help='Input folder.')
    parser.add_argument('--netG_pth_path', type=str, required=True, help='Model path.')
    parser.add_argument('--fp16_mode', type=bool, default=False, required=False)
    args = parser.parse_args()

    # init
    dm = DS_inpaint_val(args.data_input_folder)
    model = CustomTrainClass()

    # load pth generator
    model.netG.load_state_dict(torch.load(args.netG_pth_path))

    # GPU
    if args.fp16_mode == False:
      trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=20)
    # GPU with AMP (amp_level='O1' = mixed precision)
    else:
      trainer = pl.Trainer(gpus=1, precision=16, amp_level='O1', progress_bar_refresh_rate=20)
    # TPU
    #trainer = pl.Trainer(tpu_cores=8, progress_bar_refresh_rate=20)

    trainer.test(model, dm)

if __name__ == "__main__":
    main()
