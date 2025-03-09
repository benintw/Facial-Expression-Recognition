"""
Dataset handling for facial expression recognition.

This module provides:
- Custom dataset class for facial expression images
- Data augmentation pipelines using Albumentations
- Data loading utilities and DataLoader setup
- Dataset visualization tools

The dataset expects images organized in class folders.
"""

import cv2
import torch
import numpy as np
import pandas as pd


import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from icecream import ic


class FacialExpressionAlbumentationDataset(ImageFolder):

    def __init__(self, root_dir, transform=None) -> None:
        super().__init__(root=root_dir, transform=None)
        self.albumentations_transform = transform

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:

        path, label = self.samples[idx]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if self.albumentations_transform is not None:
            # img is converted from ndarray to tensor here
            augmented = self.albumentations_transform(image=img)
            img = augmented["image"]

        else:
            img = torch.from_numpy(img).type(torch.float32)

        img = img / 255.0  # torch.float32 [0, 1]

        return img, label


def get_albumentation_transforms() -> tuple[A.Compose, A.Compose]:

    train_albumentations = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ]
    )

    val_albumentations = A.Compose(
        [
            ToTensorV2(),
        ]
    )

    return train_albumentations, val_albumentations


def visualize_dataset(root_dir, augmentations=None) -> None:
    # plots augmented training img when augmentations is not None

    dataset = FacialExpressionAlbumentationDataset(root_dir, transform=augmentations)
    emotions_list = dataset.classes

    # plot the first 20 imgs (2 rows of 10)
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 3))
    fig.suptitle("Training images")
    for idx, ax in enumerate(axes.flat):
        if dataset[idx][0].dim() == 3:
            ax.imshow(dataset[idx][0].permute(1, 2, 0), cmap="magma")
        else:
            ax.imshow(dataset[idx][0], cmap="magma")
        ax.set_title(f"{emotions_list[dataset[idx][1]]}")
        ax.axis(False)

    plt.tight_layout()
    plt.show()
