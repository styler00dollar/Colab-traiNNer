from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class DFNetDataModule(pl.LightningDataModule):
    def __init__(self, training_path: str = './', validation_path: str = './', test_path: str = './', batch_size: int = 5, batch_size_DL: int = 2, num_workers: int = 2):
        super().__init__()
        self.training_dir = training_path
        self.validation_dir = validation_path
        self.test_dir = test_path
        self.batch_size = batch_size
        self.batch_size_DL = batch_size_DL
        self.num_workers = num_workers
        self.size = 256
    def setup(self, stage=None):
        img_tf = transforms.Compose([
            transforms.Resize(size=self.size),
            transforms.CenterCrop(size=self.size),
            transforms.RandomHorizontalFlip()
            #transforms.ToTensor()
        ])

        self.DFNetdataset_train = DS(self.training_dir, img_tf, self.size, batch_size_DL = self.batch_size_DL)
        self.DFNetdataset_validation = DS_green_from_mask(self.validation_dir, img_tf)
        self.DFNetdataset_test = DS_green_from_mask(self.test_dir)

    def train_dataloader(self):
        return DataLoader(self.DFNetdataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.DFNetdataset_validation, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.DFNetdataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
