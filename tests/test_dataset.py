from icecream import ic
import torch
import numpy as np
from unittest.mock import patch
from src.data.dataset import (
    get_albumentation_transforms,
    FacialExpressionAlbumentationDataset,
)

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

    def test_get_albumentation_transforms(self):
        """Test that get_albumentation_transforms returns valid transform objects."""
        train_transforms, val_transforms = get_albumentation_transforms()
        self.assertIsNotNone(train_transforms)
        self.assertIsNotNone(val_transforms)
        self.assertTrue(
            hasattr(train_transforms, "transforms")
        )  # Check for Albumentations Compose
        self.assertTrue(hasattr(val_transforms, "transforms"))

    def test_dataset_loading(self):
        """Test dataset loading with valid and invalid inputs."""
        # Test valid dataset
        dataset = FacialExpressionAlbumentationDataset(
            root_dir=self.config["data"]["train_dir"],
            transform=self.train_transforms,
        )
        self.assertGreater(len(dataset), 0)
        img, label = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.dim(), 3)  # [C, H, W] for grayscale with channel
        self.assertLessEqual(img.max(), 1.0)  # Normalized to [0, 1]
        self.assertGreaterEqual(img.min(), 0.0)
        self.assertIsInstance(label, int)

        # Test invalid directory (should raise FileNotFoundError)
        with self.assertRaises(FileNotFoundError):
            FacialExpressionAlbumentationDataset(
                root_dir="nonexistent_dir", transform=None
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

    @patch("src.data.dataset.ImageFolder")
    def test_dataset_mock(self, mock_imagefolder):
        """Test dataset with mocked ImageFolder to isolate file system."""
        mock_imagefolder.return_value = [
            (np.zeros((1, 48, 48)), 0)
        ]  # Mock grayscale image
        dataset = FacialExpressionAlbumentationDataset(
            root_dir=self.config["data"]["train_dir"],
            transform=self.train_transforms,
        )
        img, label = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, (1, 48, 48))  # Expected shape after transform
        self.assertEqual(label, 0)


if __name__ == "__main__":
    unittest.main()
