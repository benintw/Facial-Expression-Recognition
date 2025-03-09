"""
Model architecture definition for facial expression recognition.

This module implements a CNN model for facial expression classification using:
- EfficientNet backbone (via timm)
- Modified input layer for grayscale images
- Classification head for emotion categories

The model is configurable via a YAML config file.
"""

import timm
import torch
import torch.nn as nn


class FacialExpressionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.eff_net = timm.create_model(
            config["backbone"],
            pretrained=config["pretrained"],
            num_classes=config["num_classes"],
        )

        # change the conv_stem to accept grayscale images
        self.eff_net.conv_stem = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.eff_net(images)
        return logits
