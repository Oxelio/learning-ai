from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .constants import CLASS_NAMES
from .evaluation import PredictionError


def save_error_grid(
    errors: list[PredictionError],
    output_path: Path,
    *,
    count: int = 16,
    columns: int = 4,
    show: bool = True,
) -> None:
    selected_errors = sorted(
        errors,
        key=lambda error: error.confidence,
        reverse=True,
    )[:count]
    if not selected_errors:
        raise ValueError("No prediction errors are available to visualize.")
    rows = (len(selected_errors) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes = axes.flatten()
    for axis, error in zip(axes, selected_errors):
        if error.image is None:
            raise ValueError(
                "Prediction errors do not contain images. "
                "Call evaluate_classifier(..., collect_images=True)."
            )
        image = error.image.squeeze(0).numpy()
        axis.imshow(image, cmap="gray")
        axis.set_title(
            f"Expected: {CLASS_NAMES[error.expected]}\n"
            f"Predicted: {CLASS_NAMES[error.predicted]}\n"
            f"Confidence: {error.confidence * 100:.2f}%"
        )
        axis.axis("off")
    for axis in axes[len(selected_errors):]:
        axis.axis("off")
    figure.suptitle("Fashion-MNIST - Most Confident Errors", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def save_confusion_matrix(
    confusion_matrix: torch.Tensor,
    output_path: Path,
    *,
    show: bool = False,
) -> None:
    matrix = confusion_matrix.numpy()
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix)
    axis.set_title("Fashion-MNIST Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Expected")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for expected in range(len(CLASS_NAMES)):
        for predicted in range(len(CLASS_NAMES)):
            axis.text(
                predicted, expected, str(matrix[expected, predicted]),
                ha="center", va="center", fontsize=7,
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
