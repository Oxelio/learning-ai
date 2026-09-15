from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                10,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)

        return x


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

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

        loss = loss_fn(
            logits,
            labels,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total,
    )


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[
    float,
    torch.Tensor,
    list[dict],
]:
    """
    Evaluate the model and collect:

    - global accuracy
    - confusion matrix
    - misclassified samples
    """

    model.eval()

    number_of_classes = len(CLASS_NAMES)

    confusion_matrix = torch.zeros(
        number_of_classes,
        number_of_classes,
        dtype=torch.int64,
    )

    incorrect_samples = []

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

            # ----------------------------------------------------
            # Confusion matrix
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Misclassified samples
            # ----------------------------------------------------

            incorrect_mask = (
                predictions != labels
            )

            incorrect_indices = torch.nonzero(
                incorrect_mask,
                as_tuple=False,
            ).flatten()

            for index in incorrect_indices:
                expected = labels[index].item()
                predicted = predictions[index].item()

                confidence = probabilities[
                    index,
                    predicted,
                ].item()

                incorrect_samples.append(
                    {
                        "expected": expected,
                        "predicted": predicted,
                        "confidence": confidence,
                    }
                )

    accuracy = correct / total

    return (
        accuracy,
        confusion_matrix,
        incorrect_samples,
    )


def print_confusion_matrix(
    confusion_matrix: torch.Tensor,
) -> None:
    print()
    print("Confusion matrix")
    print("----------------")

    print(
        " " * 14,
        end="",
    )

    for index in range(len(CLASS_NAMES)):
        print(
            f"{index:>6}",
            end="",
        )

    print()

    for row_index, row in enumerate(
        confusion_matrix
    ):
        print(
            f"{row_index:2d} "
            f"{CLASS_NAMES[row_index][:10]:10s}",
            end="",
        )

        for value in row:
            print(
                f"{value.item():6d}",
                end="",
            )

        print()


def print_class_accuracy(
    confusion_matrix: torch.Tensor,
) -> None:
    print()
    print("Accuracy by class")
    print("-----------------")

    for class_index, class_name in enumerate(
        CLASS_NAMES
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

        print(
            f"{class_index:2d} | "
            f"{class_name:12s} | "
            f"{accuracy * 100:6.2f}% | "
            f"{correct:4d}/{total:4d}"
        )


def print_main_confusions(
    confusion_matrix: torch.Tensor,
    count: int = 10,
) -> None:
    confusions = []

    for expected in range(
        len(CLASS_NAMES)
    ):
        for predicted in range(
            len(CLASS_NAMES)
        ):
            if expected == predicted:
                continue

            occurrences = confusion_matrix[
                expected,
                predicted,
            ].item()

            if occurrences > 0:
                confusions.append(
                    (
                        occurrences,
                        expected,
                        predicted,
                    )
                )

    confusions.sort(
        reverse=True
    )

    print()
    print("Most common confusions")
    print("----------------------")

    for (
        occurrences,
        expected,
        predicted,
    ) in confusions[:count]:
        print(
            f"{CLASS_NAMES[expected]:12s} "
            f"-> "
            f"{CLASS_NAMES[predicted]:12s} "
            f": {occurrences}"
        )


def print_most_confident_errors(
    incorrect_samples: list[dict],
    count: int = 10,
) -> None:
    incorrect_samples.sort(
        key=lambda sample: sample["confidence"],
        reverse=True,
    )

    print()
    print("Most confident errors")
    print("---------------------")

    for sample in incorrect_samples[:count]:
        expected = sample["expected"]
        predicted = sample["predicted"]
        confidence = sample["confidence"]

        print(
            f"expected="
            f"{CLASS_NAMES[expected]:12s} | "
            f"predicted="
            f"{CLASS_NAMES[predicted]:12s} | "
            f"confidence="
            f"{confidence * 100:6.2f}%"
        )


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "PyTorch :",
        torch.__version__,
    )

    print(
        "Device :",
        device,
    )

    if torch.cuda.is_available():
        print(
            "GPU :",
            torch.cuda.get_device_name(0),
        )

        print(
            "HIP :",
            torch.version.hip,
        )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    transform = v2.Compose(
        [
            v2.ToImage(),

            v2.ToDtype(
                torch.float32,
                scale=True,
            ),
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root=DATA_DIRECTORY,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=DATA_DIRECTORY,
        train=False,
        download=True,
        transform=transform,
    )

    batch_size = 128

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = FashionCNN().to(
        device
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    epochs = 10

    print()
    print("Training")
    print("--------")

    for epoch in range(epochs):
        loss, accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        print(
            f"Epoch {epoch + 1:2d}/{epochs} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    (
        test_accuracy,
        confusion_matrix,
        incorrect_samples,
    ) = evaluate_model(
        model,
        test_loader,
        device,
    )

    print()
    print("Evaluation")
    print("----------")

    print(
        f"Test accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        "Correct predictions :",
        int(
            confusion_matrix.diag()
            .sum()
            .item()
        ),
    )

    print(
        "Incorrect predictions :",
        len(incorrect_samples),
    )

    # ------------------------------------------------------------
    # Detailed analysis
    # ------------------------------------------------------------

    print_confusion_matrix(
        confusion_matrix
    )

    print_class_accuracy(
        confusion_matrix
    )

    print_main_confusions(
        confusion_matrix
    )

    print_most_confident_errors(
        incorrect_samples
    )


if __name__ == "__main__":
    main()