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
from typing import Tuple

from src.data.data_loader import get_dataloaders
from src.models.model import FacialExpressionModel
from src.utils.eval_metrics import get_multiclass_accuracy
from src.utils.device import get_device
from src.utils.visualization import plot_training_history
from src.training.evaluator import Evaluator


class Trainer:
    """
    Trainer class for the facial expression recognition model.
    This class is used to train the model on the training set.
    It also keeps track of the training history.
    Args:
        config (dict): The configuration for the training.
        device_name (str): The name of the device to run the training on.
    """

    def __init__(self, config: dict, device_name: str | None = None) -> None:

        self.config = config
        self.device = get_device(device_name)
        print(f"Using device: {self.device}")
        self.setup_seeds()

        self.train_dataloader, self.val_dataloader = get_dataloaders(config)
        self.model = self._create_model(config)
        self.model.to(self.device)
        self.evaluator = Evaluator(model=self.model, config=config, device=self.device)
        self.optimizer = self._create_optimizer(self.model.parameters(), config)

        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize weights
        self._initialize_weights()

        self.grad_clip = self.config.get("grad_clip", 1.0)

        self.history: dict[str, list[float | np.ndarray]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def setup_seeds(self) -> None:
        torch.manual_seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config["random_seed"])
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.config["random_seed"])

    def _initialize_weights(self) -> None:
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _create_optimizer(self, model_params, config):
        optimizer_config = config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "Adam").lower()
        optimizer_params = optimizer_config.get(
            "params", {"lr": config["learning_rate"]}
        )

        optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        if optimizer_type not in optimizers:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_type}. Use one of {list(optimizers.keys())}"
            )
        return optimizers[optimizer_type](model_params, **optimizer_params)

    def _create_model(self, config):
        model_name = config.get("model", "FacialExpressionModel")
        models = {
            "facialexpressionmodel": FacialExpressionModel,
        }
        if model_name.lower() not in models:
            raise ValueError(
                f"Unsupported model: {model_name}. Use one of {list(models.keys())}"
            )
        return models[model_name.lower()](config)

    def train_one_epoch(self) -> tuple[float, float]:

        self.model.train()
        total_loss: float = 0.0
        total_acc: float = 0.0

        for batch in tqdm(self.train_dataloader, desc="Training"):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # logging loss and accuracy
            total_loss += loss.item()
            total_acc += get_multiclass_accuracy(logits, labels).item()

        return total_loss / len(self.train_dataloader), total_acc / len(
            self.train_dataloader
        )

    def validate(self) -> tuple[float, float]:
        avg_loss, avg_acc = self.evaluator.validate(self.val_dataloader)

        return avg_loss, avg_acc

    def train(self) -> None:

        best_val_loss: float = float("inf")
        best_val_acc: float = -float("inf")

        epochs_pbar = tqdm(
            range(self.config["epochs"]),
            desc="Training",
        )

        for epoch in epochs_pbar:
            print(f"Epoch {epoch + 1} of {self.config['epochs']:03d}")

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1

            if patience_counter >= self.config["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_training_history(
            self.history,
            save_path=Path(self.config["logging"]["log_dir"])
            / f"training_history_{timestamp}.png",
        )

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        save_path = Path(self.config["save_dir"]) / self.config["save_name"]

        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
