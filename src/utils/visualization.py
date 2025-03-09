"""
Visualization utilities for training and prediction results.

This module provides functions for visualizing:
- Training metrics (loss, accuracy)
- Model predictions
- Dataset samples with annotations
"""

import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as A
from pathlib import Path
import numpy as np


def plot_training_history(history: dict, save_path: Path | str | None = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Validation")
    ax1.set_title("Loss History")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Validation")
    ax2.set_title("Accuracy History")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
