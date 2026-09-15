from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fashion_mnist import (
    CLASS_NAMES, FashionCNN, load_external_image,
    load_model_weights, predict_image,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "fashion_cnn_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the Fashion-MNIST class of an external image."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--invert", action="store_true",
        help="Invert grayscale values for a white-background source image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionCNN().to(device)
    load_model_weights(args.checkpoint, model=model, device=device)
    image = load_external_image(args.image, invert=args.invert)
    predicted_class, confidence, probabilities = predict_image(
        model, image, device
    )

    print("Prediction :", CLASS_NAMES[predicted_class])
    print("Confidence :", f"{confidence * 100:.2f}%")
    print("\nClass probabilities\n-------------------")
    ordered = sorted(
        enumerate(probabilities.tolist()), key=lambda item: item[1], reverse=True
    )
    for class_index, probability in ordered:
        print(f"{CLASS_NAMES[class_index]:12s} : {probability * 100:6.2f}%")


if __name__ == "__main__":
    main()
