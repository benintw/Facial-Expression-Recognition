"""
Dataset visualization script for the facial expression recognition project.

This script provides visualization utilities to inspect:
- Raw training images
- Augmented training images
- Batches from the data loader
- Dataset statistics

Useful for verifying data loading and augmentation pipelines.
"""

import yaml


from src.utils.config import load_configs
from src.data.dataset import visualize_dataset
from src.data.data_loader import get_dataloaders, visualize_dataloader


def main():

    config = load_configs("configs/training.yaml")

    train_loader, valid_loader = get_dataloaders(config)

    visualize_dataset(config["train_dir"])
    visualize_dataloader(train_loader)


if __name__ == "__main__":
    main()
