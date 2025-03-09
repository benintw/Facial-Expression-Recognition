import argparse
import torch
from src.training.evaluator import Evaluator
from src.utils.config import load_configs
from src.models.model import FacialExpressionModel
from src.data.data_loader import get_dataloaders
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the model")
    parser.add_argument("--config", type=str, default="configs/validation.yaml")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["mps", "cuda", "cpu"],
        help="Device to run the training on",
    )
    args = parser.parse_args()

    config = load_configs(args.config)
    model = FacialExpressionModel(config).to(args.device)
    checkpoint_path = Path(config["save_dir"]) / config["save_name"]
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    _, val_dataloader = get_dataloaders(config)
    evaluator = Evaluator(model=model, device=args.device, config=config)
    val_loss, val_acc = evaluator.validate(val_dataloader)
    print(f"Validation loss: {val_loss}")
    print(f"Validation accuracy: {val_acc}")


if __name__ == "__main__":
    main()
