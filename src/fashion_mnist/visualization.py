from pathlib import Path

import matplotlib.pyplot as plt

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
        raise ValueError(
            "No prediction errors are available to visualize."
        )

    rows = (
        len(selected_errors)
        + columns
        - 1
    ) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3 * rows),
    )

    axes = axes.flatten()

    for axis, error in zip(
        axes,
        selected_errors,
    ):
        if error.image is None:
            raise ValueError(
                "Prediction errors do not contain images. "
                "Call evaluate_classifier(..., collect_images=True)."
            )

        image = (
            error.image
            .squeeze(0)
            .numpy()
        )

        axis.imshow(
            image,
            cmap="gray",
        )

        axis.set_title(
            f"Expected: {CLASS_NAMES[error.expected]}\n"
            f"Predicted: {CLASS_NAMES[error.predicted]}\n"
            f"Confidence: {error.confidence * 100:.2f}%"
        )

        axis.axis("off")

    for axis in axes[
        len(selected_errors):
    ]:
        axis.axis("off")

    figure.suptitle(
        "Fashion-MNIST - Most Confident Errors",
        fontsize=16,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(figure)
