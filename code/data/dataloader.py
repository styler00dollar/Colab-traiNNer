from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

class DFNetDataModule(pl.LightningDataModule):
    def __init__(self, dir_lr: str = './',  dir_hr: str = './', val_lr: str = './', val_hr: str = './', batch_size: int = 5, num_workers: int = 2, HR_size = 256, scale = 4, mask_dir: str = './', batch_size_DL=1, image_size = 256, amount_tiles = 16, canny_min = 100, canny_max = 150):
        super().__init__()

        self.dir_lr = dir_lr
        self.dir_hr = dir_hr

        self.val_lr = val_lr
        self.val_hr = val_hr

        self.batch_size = batch_size
        self.batch_size_DL = batch_size_DL

        self.num_workers = num_workers
        self.HR_size = HR_size
        self.scale = scale

        self.mask_dir = mask_dir

        self.image_size = image_size
        self.amount_tiles = amount_tiles
        self.canny_min = canny_min
        self.canny_max = canny_max

    def setup(self, stage=None):
        if cfg['datasets']['train']['mode'] == 'DS_lrhr':
          from .data import DS_lrhr, DS_lrhr_val
          self.DFNetdataset_train = DS_lrhr(self.dir_lr, self.dir_hr, self.HR_size, self.scale)
          self.DFNetdataset_validation = DS_lrhr_val(self.val_lr, self.val_hr)
          self.DFNetdataset_test = DS_lrhr_val(self.val_lr, self.val_hr)

        elif cfg['datasets']['train']['mode'] == 'DS_inpaint':
          #root, transform=None, size=256):
          from .data import DS_inpaint, DS_inpaint_val
          self.DFNetdataset_train = DS_inpaint(self.dir_hr, self.mask_dir, self.HR_size)
          self.DFNetdataset_validation = DS_inpaint_val(self.val_hr)
          self.DFNetdataset_test = DS_inpaint_val(self.val_lr)

        elif cfg['datasets']['train']['mode'] == 'DS_inpaint_tiled':
          from .data import DS_inpaint_tiled, DS_inpaint_tiled_val
          self.DFNetdataset_train = DS_inpaint_tiled(self.dir_hr, self.mask_dir, self.image_size, self.amount_tiles, self.canny_min, self.canny_max)
          self.DFNetdataset_validation = DS_inpaint_tiled_val(self.val_hr)
          self.DFNetdataset_test = DS_inpaint_tiled_val(self.val_lr)

        elif cfg['datasets']['train']['mode'] == 'DS_inpaint_tiled_batch':
          from .data import DS_inpaint_tiled_batch, DS_inpaint_tiled_batch_val
          self.DFNetdataset_train = DS_inpaint_tiled_batch(self.dir_hr, self.mask_dir, self.image_size, self.amount_tiles, self.batch_size_DL, self.canny_min, self.canny_max)
          self.DFNetdataset_validation = DS_inpaint_tiled_batch_val(self.val_hr, self.canny_min, self.canny_max)
          self.DFNetdataset_test = DS_inpaint_tiled_batch_val(self.val_lr, self.canny_min, self.canny_max)

        elif cfg['datasets']['train']['mode'] == 'DS_lrhr_batch_oft':
          from .data import DS_lrhr_batch_oft,DS_lrhr_batch_oft_val
          self.DFNetdataset_train = DS_lrhr_batch_oft(self.dir_hr, self.mask_dir, self.image_size, self.batch_size_DL, self.scale, self.image_size, self.amount_tiles)
          self.DFNetdataset_validation = DS_lrhr_batch_oft_val(self.val_hr)
          self.DFNetdataset_test = DS_lrhr_batch_oft_val(self.val_lr)

        else:
          print("Mode not found.")

    def train_dataloader(self):
        return DataLoader(self.DFNetdataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.DFNetdataset_validation, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.DFNetdataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
