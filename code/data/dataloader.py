from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .data import DS_inpaint, DS_inpaint_val, DS_lrhr, DS_lrhr_val

class DFNetDataModule(pl.LightningDataModule):
    def __init__(self, dir_lr: str = './',  dir_hr: str = './', val_lr: str = './', val_hr: str = './', batch_size: int = 5, num_workers: int = 2, hr_size = 256, scale = 4):
        super().__init__()

        self.dir_lr = dir_lr
        self.dir_hr = dir_hr

        self.val_lr = val_lr
        self.val_hr = val_hr

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hr_size = hr_size
        self.scale = scale

    def setup(self, stage=None):
        # super res
        self.DFNetdataset_train = DS_lrhr(lr_path=self.dir_lr, hr_path=self.dir_hr, hr_size = self.hr_size, scale = self.scale)
        self.DFNetdataset_validation = DS_lrhr_val(self.val_lr, self.val_hr)
        self.DFNetdataset_test = DS_lrhr_val(self.val_lr, self.val_hr)
        # inpaint
        #self.DFNetdataset_train = DS_inpaint(self.training_dir, img_tf, self.size)
        #self.DFNetdataset_validation = DS_inpaint_val(self.validation_dir, img_tf)
        #self.DFNetdataset_test = DS_inpaint_val(self.test_dir)


    def train_dataloader(self):
        return DataLoader(self.DFNetdataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.DFNetdataset_validation, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.DFNetdataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
