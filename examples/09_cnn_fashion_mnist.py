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

        # --------------------------------------------------------
        # Feature extraction
        # --------------------------------------------------------

        self.features = nn.Sequential(
            # [N, 1, 28, 28]
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            # [N, 32, 28, 28]
            # ->
            # [N, 32, 14, 14]
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # [N, 32, 14, 14]
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            # [N, 64, 14, 14]
            # ->
            # [N, 64, 7, 7]
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        # --------------------------------------------------------
        # Classification
        # --------------------------------------------------------

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


def count_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


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

        # --------------------------------------------------------
        # Forward pass
        # --------------------------------------------------------

        logits = model(images)

        # --------------------------------------------------------
        # Loss
        # --------------------------------------------------------

        loss = loss_fn(
            logits,
            labels,
        )

        # --------------------------------------------------------
        # Backpropagation
        # --------------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

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

    average_loss = (
        total_loss
        / total
    )

    accuracy = (
        correct
        / total
    )

    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
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

            loss = loss_fn(
                logits,
                labels,
            )

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

    average_loss = (
        total_loss
        / total
    )

    accuracy = (
        correct
        / total
    )

    return average_loss, accuracy


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("PyTorch :", torch.__version__)
    print("Device :", device)

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
    # Transform
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

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

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

    print()
    print("Dataset")
    print("-------")

    print(
        "Training samples :",
        len(train_dataset),
    )

    print(
        "Test samples :",
        len(test_dataset),
    )

    image, label = train_dataset[0]

    print(
        "Image shape :",
        image.shape,
    )

    print(
        "Example label :",
        label,
        "-",
        CLASS_NAMES[label],
    )

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------

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

    print()
    print("DataLoader")
    print("----------")

    print(
        "Batch size :",
        batch_size,
    )

    print(
        "Training batches :",
        len(train_loader),
    )

    print(
        "Test batches :",
        len(test_loader),
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = FashionCNN().to(device)

    print()
    print("Model")
    print("-----")

    print(model)

    print()
    print(
        "Trainable parameters :",
        f"{count_parameters(model):,}",
    )

    # ------------------------------------------------------------
    # Verify tensor dimensions
    # ------------------------------------------------------------

    example_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.features(
            example_batch
        )

        logits = model(
            example_batch
        )

    print()
    print("Tensor shapes")
    print("-------------")

    print(
        "Input :",
        example_batch.shape,
    )

    print(
        "Features :",
        features.shape,
    )

    print(
        "Output :",
        logits.shape,
    )

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------

    loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

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
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
        )

        print(
            f"Epoch {epoch + 1:2d}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test: {test_accuracy * 100:.2f}%"
        )

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    print()
    print("Predictions")
    print("-----------")

    model.eval()

    with torch.no_grad():
        for index in range(10):
            image, label = test_dataset[index]

            input_image = (
                image
                .unsqueeze(0)
                .to(device)
            )

            logits = model(
                input_image
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_class = (
                probabilities
                .argmax(dim=1)
                .item()
            )

            confidence = (
                probabilities[
                    0,
                    predicted_class,
                ]
                .item()
            )

            print(
                f"{index:2d} | "
                f"expected={CLASS_NAMES[label]:12s} | "
                f"predicted={CLASS_NAMES[predicted_class]:12s} | "
                f"confidence={confidence * 100:.2f}%"
            )


if __name__ == "__main__":
    main()