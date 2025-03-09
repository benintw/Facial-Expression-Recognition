import unittest
import yaml
import torch
from icecream import ic
from src.models.model import FacialExpressionModel


class TestFacialExpressionModel(unittest.TestCase):
    def setUp(self):
        """Load the configuration file for all tests."""
        with open("configs/model.yaml", "r") as f:
            self.config = yaml.safe_load(f)

    def test_model_init(self):
        """Test model initialization with a valid config."""

        # Create a new model instance for this test
        model = FacialExpressionModel(self.config)
        self.assertIsNotNone(model)

        # Add more specific assertions about the model's state
        self.assertTrue(hasattr(model, "forward"))  # Check if model has forward method

    def test_model_init_invalid_config(self):
        """Test model initialization with an invalid config."""

        # Test with a bad config (e.g., missing required keys)
        invalid_config = {"wrong_key": "value"}

        with self.assertRaises(KeyError):  # Adjust exception type as needed
            FacialExpressionModel(invalid_config)

    def test_model_forward(self):
        """Test the model's forward pass with a mock input."""
        # Create a new model instance for this test
        model = FacialExpressionModel(self.config)

        # Mock input: batch of 5 grayscale images (1 channel, 48x48)
        images = torch.randn(5, 1, 48, 48)

        # Forward pass
        logits = model(images)

        # Validate output
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (5, 7))  # Assuming 7 emotion classes
        self.assertFalse(torch.isnan(logits).any())  # Check for NaN values
        ic(logits.shape)  # Keep for debugging

    def test_model_forward_invalid_input(self):
        """Test the forward pass with invalid input dimensions."""
        model = FacialExpressionModel(self.config)
        # Invalid input: wrong number of channels (e.g., RGB instead of grayscale)
        invalid_images = torch.randn(5, 3, 48, 48)
        with self.assertRaises(RuntimeError):  # Adjust based on expected error
            model(invalid_images)


if __name__ == "__main__":
    unittest.main()
