"""
22-Dez-21
https://github.com/adam-mehdi/MuarAugment/blob/02fa7a98796fba801403bf8df28d64ffd399f8e6/muar/augmentations.py
https://github.com/adam-mehdi/MuarAugment/blob/02fa7a98796fba801403bf8df28d64ffd399f8e6/muar/loss.py
https://github.com/adam-mehdi/MuarAugment/blob/02fa7a98796fba801403bf8df28d64ffd399f8e6/muar/transform_lists.py
"""
import albumentations as A
import kornia.augmentation as K

################################################ albumentations_list ###############################


def albumentations_list(MAGN: int = 4):
    """
    Returns standard list of albumentations transforms, each of mangitude `MAGN`.

    Args:
        MAGN (int): Magnitude of each transform in the returned list.
    """
    M = MAGN
    transform_list = [
        # PIXEL-LEVEL
        A.RandomContrast(limit=M * 0.1, always_apply=True),
        A.RandomBrightness(limit=M * 0.1, always_apply=True),
        A.Equalize(always_apply=True),
        A.OpticalDistortion(
            distort_limit=M * 0.2, shift_limit=M * 0.1, always_apply=True
        ),
        A.RGBShift(
            r_shift_limit=M * 10,
            g_shift_limit=M * 10,
            b_shift_limit=M * 10,
            always_apply=True,
        ),
        A.ISONoise(
            color_shift=(M * 0.01, M * 0.1),
            intensity=(M * 0.02, M * 0.2),
            always_apply=True,
        ),
        A.RandomFog(fog_coef_lower=M * 0.01, fog_coef_upper=M * 0.1, always_apply=True),
        A.CoarseDropout(max_holes=M * 10, always_apply=True),
        A.GaussNoise(var_limit=(M, M * 50), always_apply=True),
        # SPATIAL
        A.Rotate(always_apply=True),
        A.Transpose(always_apply=True),
        A.NoOp(always_apply=True),
        A.ElasticTransform(
            alpha=M * 0.25, sigma=M * 3, alpha_affine=M * 3, always_apply=True
        ),
        A.GridDistortion(distort_limit=M * 0.075, always_apply=True),
    ]
    return transform_list


######################################################## kornia_list #############################


def kornia_list(MAGN: int = 4):
    """
    Returns standard list of kornia transforms, each with magnitude `MAGN`.

    Args:
        MAGN (int): Magnitude of each transform in the returned list.
    """
    transform_list = [
        # SPATIAL
        K.RandomHorizontalFlip(p=1),
        K.RandomRotation(degrees=90.0, p=1),
        K.RandomAffine(degrees=MAGN * 5.0, shear=MAGN / 5, translate=MAGN / 20, p=1),
        K.RandomPerspective(distortion_scale=MAGN / 25, p=1),
        # PIXEL-LEVEL
        K.ColorJitter(brightness=MAGN / 30, p=1),  # brightness
        K.ColorJitter(saturation=MAGN / 30, p=1),  # saturation
        K.ColorJitter(contrast=MAGN / 30, p=1),  # contrast
        K.ColorJitter(hue=MAGN / 30, p=1),  # hue
        K.ColorJitter(p=0),  # identity
        K.RandomMotionBlur(
            kernel_size=2 * (MAGN // 3) + 1, angle=MAGN, direction=1.0, p=1
        ),
        K.RandomErasing(scale=(MAGN / 100, MAGN / 50), ratio=(MAGN / 20, MAGN), p=1),
    ]
    return transform_list


from typing import Union
from random import random

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

import kornia.augmentation as K


###################################### BatchRandAugment ###########################################


class BatchRandAugment(nn.Module):
    """
    Image augmentation pipeline that applies a composition of `N_TFMS` transforms
    each of magnitude `MAGN` sampled uniformly at random from `transform_list` with
    optional batch resizing and label mixing transforms.

    Args:
        N_TFMS (int): Number of transformations sampled for each composition,
                      excluding resize or label mixing transforms.      <N in paper>
        MAGN (int): Magnitude of augmentation applied. Ranges from [0, 10] with
                    10 being the max magnitude.                         <M in paper>
        mean (tuple, torch.Tensor): Mean of images after normalized in range [0,1]
        std (tuple, torch.Tensor): Mean of images after normalized in range [0,1]
        transform_list (list): List of transforms to sample from. Default list
                               provided if not specified.
        use_resize (int): Batch-wise resize transform to apply. Options:
            None: Don't use.
            0: RandomResizedCrop
            1: RandomCrop
            2: CenterCrop
            3: Randomly select a resize transform per batch.
        image_size (tuple): Final size after applying batch-wise resize transforms.
        use_mix (int): Label mixing transform to apply. Options:
            None: No label mixing transforms.
            1: MixUp
        mix_p (float): probability of applying the mix transform on a batch
                       given `use_mix` is not None.
    """

    def __init__(
        self,
        N_TFMS: int,
        MAGN: int,
        mean: Union[tuple, list, torch.tensor],
        std: Union[tuple, list, torch.tensor],
        transform_list: list = None,
        use_resize: int = None,
        image_size: tuple = None,
        use_mix: int = None,
        mix_p: float = 0.5,
    ):
        super().__init__()

        self.N_TFMS, self.MAGN = N_TFMS, MAGN
        self.use_mix, self.mix_p = use_mix, mix_p
        self.image_size = image_size

        if not isinstance(mean, torch.Tensor):
            mean = torch.Tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.Tensor(std)

        if self.use_mix is not None:
            self.mix_list = [
                K.RandomCutMix(self.image_size[0], self.image_size[1], p=1),
                K.RandomMixUp(p=1),
            ]

        self.use_resize = use_resize
        if use_resize is not None:
            assert (
                len(image_size) == 2
            ), "Invalid `image_size`. Must be a tuple of form (h, w)"
            self.resize_list = [
                K.RandomResizedCrop(image_size),
                K.RandomCrop(image_size),
                K.CenterCrop(image_size),
            ]
            if self.use_resize < 3:
                self.resize = self.resize_list[use_resize]

        self.normalize = K.Normalize(mean, std)

        self.transform_list = transform_list
        if transform_list is None:
            self.transform_list = kornia_list(MAGN)

    def setup(self):
        if self.use_resize == 3:
            self.resize = np.random.choice(self.resize_list)

        if self.use_mix is not None and random() < self.mix_p:
            self.mix = self.mix_list[self.use_mix]
        else:
            self.mix = None

        sampled_tfms = list(
            np.random.choice(self.transform_list, self.N_TFMS, replace=False)
        ) + [self.normalize]
        self.transform = nn.Sequential(*sampled_tfms)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Applies transforms on the batch.

        If `use_mix` is not `None`, `y` is required. Else, it is optional.

        If a label-mixing transform is applied on the batch, `y` is returned
        in shape `(batch_size, 3)`, in which case use the special loss function
        provided in `muar.utils` if.

        Args:
            x (torch.Tensor): Batch of input images.
            y (torch.Tensor): Batch of labels.
        """
        if self.use_resize is not None:
            x = self.resize(x)

        if self.mix is not None:
            x, y = self.mix(x, y)
            return self.transform(x), y

        elif y is None:
            return self.transform(x)

        return self.transform(x), y


####################################### AlbumentationsRandAugment ####################################


class AlbumentationsRandAugment:
    """
    Item-wise RandAugment using Albumentations transforms. Use this to apply
    RandAugment within `Dataset`.
    """

    def __init__(self, N_TFMS: int = 2, MAGN: int = 4, transform_list: list = None):
        """
        Args:
            N_TFMS (int): Number of transformation in each composition.
            MAGN (int): Magnitude of augmentation applied.
            tranform: List of K transformations to sample from.
        """

        if transform_list is None:
            transform_list = albumentations_list(MAGN)

        self.transform_list = transform_list
        self.MAGN = MAGN
        self.N_TFMS = N_TFMS

    def __call__(self):
        """
        Returns a randomly sampled list of `N_TFMS` transforms from `transform_list`
        (default list provided if `None`).
        """
        sampled_tfms = np.random.choice(self.transform_list, self.N_TFMS)
        return list(sampled_tfms)


############################################### MuAugment ############################################


class MuAugment:
    """
    For each image in the batch, applies `N_COMPS` compositions with `N_TFMS`
    tranformations, each of magnitude `MAGN`, and for each group of `N_COMPS`
    augmented images, returns the `N_SELECTED` images with the greatest loss.
    """

    def __init__(
        self,
        rand_augment: BatchRandAugment,
        N_COMPS: int = 4,
        N_SELECTED: int = 2,
        device: Union[torch.device, str] = None,
    ):
        """
        Args:
            rand_augment (muar.BatchRandAugment): RandAugment pipeline to apply.
            N_COMPS (int): Number of compositions placed on each image. [C in paper]
            N_SELECTED (int): Number of selected compositions for each image. [S in paper]
            device (torch.device, optional): Device to use: cuda or cpu. Inferred if `None`.

        Returns:
            S_images (torch.Tensor): (batch_size*S, n_channels, height, width)
            S_targets (torch.LongTensor): (batch_size*N_SELECTED) or
                                          (batch_size*N_SELECTED, 3) if applied MixUp or CutMix

        NOTE: Effective batch size is batch_size*S. Increasing C yields better accuracy.
        """
        super().__init__()

        self.rand_augment = rand_augment
        self.C, self.S = N_COMPS, N_SELECTED

        self.device = device
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(
        self,  # TODO: ALLOW SCHEDULING OF HYPERPARAMETERS (EDIT THEM VIA KWARGS)
        model: Union[nn.Module, pl.LightningModule],
    ):
        """
        Args:
            model (nn.Module, pl.LightningModule): Model used to calculate loss.
        """
        self.model = model
        self.rand_augment.setup()

        self.transform = self.rand_augment
        self.N, self.M = self.rand_augment.N_TFMS, self.rand_augment.MAGN
        self.image_size = self.rand_augment.image_size

    @torch.no_grad()
    def __call__(self, batch):
        """
        Args:
            batch (tuple): tuple of (image batch, target batch).
                image batch: (batch_size, n_channels, height, width)
                target batch: (batch_size, 1) or (batch_size)
        """
        xb, yb = batch
        if xb.device != self.device:
            xb = xb.to(self.device)
        if yb.device != self.device:
            yb = yb.to(self.device)

        if len(yb.shape) == 2:
            yb = yb.squeeze(1)
        BS, N_CHANNELS, HEIGHT, WIDTH = xb.shape
        if self.image_size is not None:
            HEIGHT, WIDTH = self.image_size
        C_images = torch.zeros(
            self.C, BS, N_CHANNELS, HEIGHT, WIDTH, device=self.device
        )

        if self.transform.mix is not None:
            C_targets = torch.zeros(self.C, BS, 3, device=self.device, dtype=torch.long)
        else:
            C_targets = torch.zeros(self.C, BS, device=self.device, dtype=torch.long)

        for c in range(self.C):
            xbt, ybt = self.transform(xb, yb)
            C_images[c], C_targets[c] = xbt, ybt

        # is only one class, it is not possible to calculate loss like muaraugment does, creating a list of zeros with batch size
        S_idxs = [0] * yb.shape[0]

        S_images = C_images[S_idxs, range(BS)]
        S_images = S_images.view(-1, N_CHANNELS, HEIGHT, WIDTH)

        S_targets = C_targets[S_idxs, range(BS)]
        S_targets = S_targets.view(-1, 3) if self.transform.mix else S_targets.view(-1)

        return S_images, S_targets
