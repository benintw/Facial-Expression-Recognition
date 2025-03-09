"""
Training script for the facial expression recognition model.

This script handles the complete training pipeline including:
- Model initialization
- Data loading
- Training loop with validation
- Checkpointing
- Early stopping
- Training visualization

The training parameters are configured via a YAML config file.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader
from src.utils.eval_metrics import get_multiclass_accuracy


class Evaluator:
    """
    Evaluator class for the facial expression recognition model.
    This class is used to evaluate the model on the validation set.
    It also keeps track of the training history.
    Args:
        model (nn.Module): The model to evaluate.
        config (dict): The configuration for the training.
        device (torch.device): The device to run the evaluation on.

    """

    def __init__(self, *, model: nn.Module, config: dict, device: torch.device) -> None:

        self.model = model
        self.config = config
        self.device = device
        print(f"Using device: {self.device}")

        self.loss_fn = nn.CrossEntropyLoss()

        self.history: dict[str, list[float | np.ndarray]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def validate(self, dataloader: DataLoader) -> tuple[float, float]:
        """
        Validate the model on the validation set.
        Args:
            dataloader (DataLoader): The validation dataloader.
        Returns:
            tuple[float, float]: The average loss and accuracy of the model on the validation set.
        """
        self.model.eval()
        total_loss: float = 0.0
        total_acc: float = 0.0

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Validation"):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

                total_loss += loss.item()
                total_acc += get_multiclass_accuracy(logits, labels).item()

        return total_loss / len(dataloader), total_acc / len(dataloader)
