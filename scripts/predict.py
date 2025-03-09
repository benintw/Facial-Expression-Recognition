def main() -> None:
    import argparse
    from src.inference.predictor import Predictor
    from src.utils.config import load_configs
    from src.models.model import FacialExpressionModel

    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--input", type=str, default="images/test_image.jpg")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for prediction",
    )

    args = parser.parse_args()

    config = load_configs(args.config)

    model = FacialExpressionModel(config).to(args.device)
    predictor = Predictor(model=model, config=config, device=args.device)
    original_image, probs, pred_class = predictor.predict(args.input)
    predictor.save_predictions(original_image, probs)


if __name__ == "__main__":
    main()
