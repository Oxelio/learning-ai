from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fashion_mnist import (
    CLASS_NAMES, FashionCNN, class_accuracies, create_dataloaders,
    create_datasets, evaluate_classifier, load_model_weights,
    most_common_confusions,
)
from fashion_mnist.visualization import save_confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "fashion_cnn_best.pt"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "evaluation.json"
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "outputs" / "confusion_matrix.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Fashion-MNIST CNN."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionCNN().to(device)
    metadata = load_model_weights(args.checkpoint, model=model, device=device)
    _, test_dataset = create_datasets()
    _, test_loader = create_dataloaders(
        test_dataset, test_dataset, batch_size=128, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    result = evaluate_classifier(model, test_loader, device)
    per_class_accuracy = class_accuracies(result.confusion_matrix)
    confusions = most_common_confusions(result.confusion_matrix, count=10)

    print("Checkpoint :", args.checkpoint)
    if metadata:
        print("Checkpoint epoch :", metadata.get("epoch", "?"))
        print("Validation loss :", metadata.get("validation_loss", "?"))
    print("\nTest accuracy :", f"{result.accuracy * 100:.2f}%")

    print("\nAccuracy by class\n-----------------")
    for index, accuracy in enumerate(per_class_accuracy):
        print(f"{index:2d} | {CLASS_NAMES[index]:12s} | {accuracy * 100:6.2f}%")

    print("\nMost common confusions\n----------------------")
    for occurrences, expected, predicted in confusions:
        print(
            f"{CLASS_NAMES[expected]:12s} -> "
            f"{CLASS_NAMES[predicted]:12s} : {occurrences}"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as file:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": metadata.get("epoch"),
            "validation_loss": metadata.get("validation_loss"),
            "test_accuracy": result.accuracy,
            "class_accuracy": {
                name: accuracy for name, accuracy in zip(CLASS_NAMES, per_class_accuracy)
            },
            "most_common_confusions": [
                {
                    "expected": CLASS_NAMES[expected],
                    "predicted": CLASS_NAMES[predicted],
                    "count": occurrences,
                }
                for occurrences, expected, predicted in confusions
            ],
        }, file, indent=2)

    save_confusion_matrix(result.confusion_matrix, args.matrix, show=False)
    print("\nArtifacts\n---------")
    print("Report :", args.report)
    print("Matrix :", args.matrix)


if __name__ == "__main__":
    main()
