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
from src.training.trainer import Trainer
from src.utils.config import load_configs


def main() -> None:

    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["mps", "cuda", "cpu"],
        help="Device to run the training on",
    )
    args = parser.parse_args()

    config = load_configs(args.config)
    trainer = Trainer(config=config, device_name=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
