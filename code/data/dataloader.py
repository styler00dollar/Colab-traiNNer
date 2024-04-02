from torch.utils.data import DataLoader
import pytorch_lightning as pl

import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dir_lr: str = "./",
        dir_hr: str = "./",
        val_lr: str = "./",
        val_hr: str = "./",
        batch_size: int = 5,
        num_workers: int = 2,
        HR_size=256,
        scale=4,
        mask_dir: str = "./",
        image_size=256,
        amount_tiles=16,
        canny_min=100,
        canny_max=150,
    ):
        super().__init__()

        self.dir_lr = dir_lr
        self.dir_hr = dir_hr

        self.val_lr = val_lr
        self.val_hr = val_hr

        self.batch_size = batch_size

        self.num_workers = num_workers
        self.HR_size = HR_size
        self.scale = scale

        self.mask_dir = mask_dir

        self.image_size = image_size
        self.amount_tiles = amount_tiles
        self.canny_min = canny_min
        self.canny_max = canny_max

    def setup(self, stage=None):
        if cfg["datasets"]["train"]["mode"] == "DS_lrhr":
            from .data import DS_lrhr, DS_lrhr_val

            self.dataset_train = DS_lrhr(
                self.dir_lr, self.dir_hr, self.HR_size, self.scale
            )
            self.dataset_validation = DS_lrhr_val(self.val_lr, self.val_hr)
            self.dataset_test = DS_lrhr_val(self.val_lr, self.val_hr)

        elif cfg["datasets"]["train"]["mode"] == "DS_inpaint":
            # root, transform=None, size=256):
            from .data import DS_inpaint, DS_inpaint_val

            self.dataset_train = DS_inpaint(self.dir_hr, self.mask_dir, self.HR_size)
            self.dataset_validation = DS_inpaint_val(self.val_hr)
            self.dataset_test = DS_inpaint_val(self.val_lr)

        elif cfg["datasets"]["train"]["mode"] == "DS_fontgen":
            from .data import DS_fontgen, DS_fontgen_val

            self.dataset_train = DS_fontgen(self.dir_hr)
            self.dataset_validation = DS_fontgen_val(self.val_lr, self.val_hr)
            self.dataset_test = DS_fontgen_val(self.val_lr, self.val_hr)

        elif cfg["datasets"]["train"]["mode"] == "DS_video":
            from .data_video import VimeoTriplet, VimeoTriplet_val

            self.dataset_train = VimeoTriplet(self.dir_hr)
            self.dataset_validation = VimeoTriplet_val(self.val_hr)
            self.dataset_test = VimeoTriplet_val(self.val_hr)

        elif cfg["datasets"]["train"]["mode"] == "DS_video_direct":
            from .data_video import VimeoTripletDirect, VimeoTriplet_val

            self.dataset_train = VimeoTripletDirect(self.dir_hr)
            self.dataset_validation = VimeoTriplet_val(self.val_hr)
            self.dataset_test = VimeoTriplet_val(self.val_hr)

        elif cfg["datasets"]["train"]["mode"] == "DS_inpaint_TF":
            from .data import DS_inpaint_TF, DS_inpaint_val

            self.dataset_train = DS_inpaint_TF()
            self.dataset_validation = DS_inpaint_val(self.val_hr)
            self.dataset_test = DS_inpaint_val(self.val_lr)

        elif cfg["datasets"]["train"]["mode"] == "DS_svg_TF":
            from .data import DS_svg_TF, DS_lrhr_val

            self.dataset_train = DS_svg_TF()
            self.dataset_validation = DS_lrhr_val(self.val_lr, self.val_hr)
            self.dataset_test = DS_lrhr_val(self.val_lr, self.val_hr)

        elif cfg["datasets"]["train"]["mode"] == "DS_realesrgan":
            from .data import DS_lrhr_val
            from .realesrgan import RealESRGANDataset

            self.dataset_train = RealESRGANDataset(
                self.dir_hr, self.HR_size, self.scale
            )
            self.dataset_validation = DS_lrhr_val(self.val_lr, self.val_hr)
            self.dataset_test = DS_lrhr_val(self.val_lr, self.val_hr)

        else:
            print("Mode not found.")

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_validation, batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
