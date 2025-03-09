"""
Script for making predictions using a trained facial expression recognition model.

This script loads a trained model and uses it to predict facial expressions for input images.
It includes functionality for preprocessing images, making predictions, and visualizing results.

Example usage:
    python -m facial_expression_recognition.scripts.predict --config configs/predict_config.yaml --input data/test_img.jpg
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import torch.nn as nn

from icecream import ic
import matplotlib.pyplot as plt


class Predictor:
    def __init__(self, *, model: nn.Module, config: dict, device: torch.device) -> None:
        self.config = config
        self.device = device
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.model.eval()

        self.load_checkpoint()

    def load_checkpoint(self) -> None:
        checkpoint_path = Path(self.config["save_dir"]) / self.config["save_name"]

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def _preprocess_image(self, original_img_np: np.ndarray) -> torch.Tensor:
        if original_img_np is None or original_img_np.size == 0:
            raise ValueError("Input image is empty or None")

        img_np = np.expand_dims(original_img_np, axis=-1)

        image_tensor: torch.Tensor = (torch.from_numpy(img_np).permute(2, 0, 1)).type(
            torch.float32
        )
        image_tensor = image_tensor / 255.0
        return image_tensor.unsqueeze(dim=0)

    def predict(self, image_path: str) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:

        self.model.eval()

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        original_image = cv2.imread(image_path)  # (H,W,C)
        if original_image is None:
            raise ValueError(f"Failed to read image from {image_path}")

        original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        input_tensor = self._preprocess_image(original_image_gray)

        with torch.inference_mode():
            logits = self.model(input_tensor.to(self.device))

        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1)

        return original_image, probs, pred_class

    def save_predictions(
        self,
        original_image: np.ndarray,  # (H,W,C)
        probs: torch.Tensor,  # (1,7)
    ) -> None:

        classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

        save_dir = Path(self.config["prediction"]["output_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "predictions.png"

        probs_np = probs.cpu().squeeze().numpy()

        fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), ncols=2)
        ax1.imshow(original_image)
        ax1.axis("off")
        ax1.set_title(f"{classes[np.argmax(probs_np)]}")

        ax2.barh(classes, probs_np)
        ax2.set_aspect(0.1)
        ax2.set_yticks(classes)
        ax2.set_yticklabels(classes)
        ax2.set_title("Class Probability")
        ax2.set_xlim(0, 1.1)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
