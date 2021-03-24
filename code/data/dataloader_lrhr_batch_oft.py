from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class DFNetDataModule(pl.LightningDataModule):
    def __init__(self, training_path: str = './', val_lr: str = './', val_hr: str = './', batch_size: int = 5, batch_size_DL: int = 2, num_workers: int = 2, hr_size=256, scale = 4, image_size = 400, amount_tiles=3):
        super().__init__()
        self.training_dir = training_path
        self.val_lr = val_lr
        self.val_hr = val_hr

        #self.test_dir = test_path
        self.batch_size = batch_size
        self.batch_size_DL = batch_size_DL
        self.num_workers = num_workers
        self.hr_size = hr_size
        self.scale = scale
        self.image_size = image_size
        self.amount_tiles = amount_tiles

    def setup(self, stage=None):
        self.DFNetdataset_train = DS(self.training_dir, self.hr_size, batch_size_DL = self.batch_size_DL, scale=self.scale, image_size = self.image_size, amount_tiles = self.amount_tiles)
        self.DFNetdataset_validation = DS_val(self.val_lr, self.val_hr)
        self.DFNetdataset_test = DS_val(self.val_lr, self.val_hr)

    def train_dataloader(self):
        return DataLoader(self.DFNetdataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.DFNetdataset_validation, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.DFNetdataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
