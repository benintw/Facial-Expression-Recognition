from icecream import ic
import torch
from unittest.mock import patch
from src.data.dataset import (
    get_albumentation_transforms,
    FacialExpressionAlbumentationDataset,
)
from src.data.data_loader import get_dataloaders
import unittest


class TestData(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures: config, transforms, and mock dataset."""
        self.config = {
            "data": {
                "train_dir": "data/train",  # Ensure this directory exists with sample images
                "val_dir": "data/validation",
                "batch_size": 64,
                "num_workers": 0,  # Set to 0 for testing to avoid multiprocessing issues
            }
        }
        self.train_transforms, self.val_transforms = get_albumentation_transforms()
        self.train_dataset = FacialExpressionAlbumentationDataset(
            root_dir=self.config["data"]["train_dir"],
            transform=self.train_transforms,
        )
        self.valid_dataset = FacialExpressionAlbumentationDataset(
            root_dir=self.config["data"]["val_dir"],
            transform=self.val_transforms,
        )

    def test_get_dataloaders(self):
        """Test that get_dataloaders returns functional DataLoader objects."""
        train_loader, valid_loader = get_dataloaders(self.config)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(valid_loader)
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(valid_loader, torch.utils.data.DataLoader)

        # Test batch loading
        train_batch = next(iter(train_loader))
        images, labels = train_batch
        self.assertIsInstance(images, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(
            images.shape[0], self.config["data"]["batch_size"]
        )  # Batch size
        self.assertEqual(images.shape[1], 1)  # Grayscale channel
        self.assertGreater(images.shape[2], 0)  # Height
        self.assertGreater(images.shape[3], 0)  # Width
        self.assertEqual(labels.shape[0], self.config["data"]["batch_size"])


if __name__ == "__main__":
    unittest.main()
