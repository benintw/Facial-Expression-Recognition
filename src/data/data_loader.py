"""
Dataset handling for facial expression recognition.

This module provides:
- Custom dataset class for facial expression images
- Data augmentation pipelines using Albumentations
- Data loading utilities and DataLoader setup
- Dataset visualization tools

The dataset expects images organized in class folders.
"""

import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from icecream import ic

from src.data.dataset import (
    FacialExpressionAlbumentationDataset,
    get_albumentation_transforms,
)


def get_dataloaders(config) -> tuple[DataLoader, DataLoader]:
    train_albumentations, val_albumentations = get_albumentation_transforms()
    train_dataset = FacialExpressionAlbumentationDataset(
        config["train_dir"], train_albumentations
    )
    valid_dataset = FacialExpressionAlbumentationDataset(
        config["val_dir"], val_albumentations
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    return train_loader, valid_loader


def visualize_dataloader(dataloader) -> None:
    images, labels = next(iter(dataloader))

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Images from the dataloader")
    plt.imshow(
        torchvision.utils.make_grid(images[:64], padding=2, normalize=True).permute(
            1, 2, 0
        )
    )
    plt.show()
