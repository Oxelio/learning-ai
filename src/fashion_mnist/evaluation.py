from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from .constants import CLASS_NAMES


@dataclass(slots=True)
class PredictionError:
    expected: int
    predicted: int
    confidence: float
    expected_probability: float
    image: torch.Tensor | None = None


@dataclass(slots=True)
class EvaluationResult:
    accuracy: float
    confusion_matrix: torch.Tensor
    errors: list[PredictionError]


def evaluate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    collect_images: bool = False,
) -> EvaluationResult:
    model.eval()

    number_of_classes = len(CLASS_NAMES)

    confusion_matrix = torch.zeros(
        number_of_classes,
        number_of_classes,
        dtype=torch.int64,
    )

    errors: list[PredictionError] = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = probabilities.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            encoded = (
                labels * number_of_classes
                + predictions
            )

            batch_matrix = torch.bincount(
                encoded,
                minlength=(
                    number_of_classes
                    * number_of_classes
                ),
            ).reshape(
                number_of_classes,
                number_of_classes,
            )

            confusion_matrix += batch_matrix.cpu()

            incorrect_indices = torch.nonzero(
                predictions != labels,
                as_tuple=False,
            ).flatten()

            for index in incorrect_indices:
                expected = int(labels[index].item())
                predicted = int(
                    predictions[index].item()
                )

                image = None

                if collect_images:
                    image = (
                        images[index]
                        .detach()
                        .cpu()
                    )

                errors.append(
                    PredictionError(
                        expected=expected,
                        predicted=predicted,
                        confidence=float(
                            probabilities[
                                index,
                                predicted,
                            ].item()
                        ),
                        expected_probability=float(
                            probabilities[
                                index,
                                expected,
                            ].item()
                        ),
                        image=image,
                    )
                )

    return EvaluationResult(
        accuracy=correct / total,
        confusion_matrix=confusion_matrix,
        errors=errors,
    )


def class_accuracies(
    confusion_matrix: torch.Tensor,
) -> list[float]:
    accuracies: list[float] = []

    for class_index in range(
        len(CLASS_NAMES)
    ):
        correct = confusion_matrix[
            class_index,
            class_index,
        ].item()

        total = confusion_matrix[
            class_index
        ].sum().item()

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        accuracies.append(accuracy)

    return accuracies


def most_common_confusions(
    confusion_matrix: torch.Tensor,
    *,
    count: int = 10,
) -> list[tuple[int, int, int]]:
    confusions: list[
        tuple[int, int, int]
    ] = []

    for expected in range(
        len(CLASS_NAMES)
    ):
        for predicted in range(
            len(CLASS_NAMES)
        ):
            if expected == predicted:
                continue

            occurrences = int(
                confusion_matrix[
                    expected,
                    predicted,
                ].item()
            )

            if occurrences > 0:
                confusions.append(
                    (
                        occurrences,
                        expected,
                        predicted,
                    )
                )

    confusions.sort(reverse=True)

    return confusions[:count]
