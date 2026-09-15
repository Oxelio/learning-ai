from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = PROJECT_ROOT / "data"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"

OUTPUT_PATH = OUTPUT_DIRECTORY / "fashion_mnist_errors.png"


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


def collect_errors(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> list[dict]:
    model.eval()

    errors = []

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

                expected_probability = probabilities[
                    index,
                    expected,
                ].item()

                errors.append(
                    {
                        "image": images[index]
                        .detach()
                        .cpu(),

                        "expected": expected,
                        "predicted": predicted,

                        "confidence": confidence,

                        "expected_probability":
                            expected_probability,
                    }
                )

    return errors


def visualize_errors(
    errors: list[dict],
    count: int = 16,
) -> None:
    errors = sorted(
        errors,
        key=lambda error: error["confidence"],
        reverse=True,
    )

    selected_errors = errors[:count]

    columns = 4

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
        image = (
            error["image"]
            .squeeze(0)
            .numpy()
        )

        expected = error["expected"]
        predicted = error["predicted"]

        confidence = error["confidence"]

        axis.imshow(
            image,
            cmap="gray",
        )

        axis.set_title(
            f"Expected: {CLASS_NAMES[expected]}\n"
            f"Predicted: {CLASS_NAMES[predicted]}\n"
            f"Confidence: {confidence * 100:.2f}%"
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

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    print()
    print(
        "Visualization saved :",
        OUTPUT_PATH,
    )

    plt.show()


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
    # Error analysis
    # ------------------------------------------------------------

    errors = collect_errors(
        model,
        test_loader,
        device,
    )

    print()
    print("Errors")
    print("------")

    print(
        "Misclassified samples :",
        len(errors),
    )

    errors.sort(
        key=lambda error: error["confidence"],
        reverse=True,
    )

    print()
    print("Most confident errors")
    print("---------------------")

    for error in errors[:10]:
        print(
            f"expected="
            f"{CLASS_NAMES[error['expected']]:12s} | "
            f"predicted="
            f"{CLASS_NAMES[error['predicted']]:12s} | "
            f"predicted confidence="
            f"{error['confidence'] * 100:6.2f}% | "
            f"expected probability="
            f"{error['expected_probability'] * 100:6.2f}%"
        )

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------

    visualize_errors(
        errors,
        count=16,
    )


if __name__ == "__main__":
    main()