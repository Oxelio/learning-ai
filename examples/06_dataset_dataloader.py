import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, random_split


def create_dataset(
    size: int,
) -> TensorDataset:
    """
    Create a synthetic XOR-like binary classification dataset.

    Class 1: x and y have the same sign.
    Class 0: x and y have different signs.
    """

    inputs = torch.rand(size, 2) * 2 - 1

    labels = (
        inputs[:, 0] * inputs[:, 1] > 0
    ).float().unsqueeze(1)

    return TensorDataset(
        inputs,
        labels,
    )


def calculate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(inputs)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.numel()

    return correct / total


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
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
    # Dataset
    # ------------------------------------------------------------

    dataset = create_dataset(2000)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )

    print()
    print("Dataset")
    print("-------")

    print(
        "Total samples :",
        len(dataset),
    )

    print(
        "Training samples :",
        len(train_dataset),
    )

    print(
        "Test samples :",
        len(test_dataset),
    )

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------

    batch_size = 64

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

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),

        nn.Linear(16, 16),
        nn.ReLU(),

        nn.Linear(16, 1),
    ).to(device)

    print()
    print("Model")
    print("-----")
    print(model)

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------

    loss_fn = nn.BCEWithLogitsLoss()

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

    epochs = 100

    print()
    print("Training")
    print("--------")

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for inputs, labels in train_loader:
            # Move only the current batch to the GPU.
            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            # Forward pass
            logits = model(inputs)

            # Loss
            loss = loss_fn(
                logits,
                labels,
            )

            # Reset previous gradients
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Update parameters
            optimizer.step()

            running_loss += (
                loss.item()
                * inputs.size(0)
            )

        epoch_loss = (
            running_loss
            / len(train_dataset)
        )

        if epoch % 10 == 0:
            train_accuracy = calculate_accuracy(
                model,
                train_loader,
                device,
            )

            test_accuracy = calculate_accuracy(
                model,
                test_loader,
                device,
            )

            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {epoch_loss:.6f} | "
                f"Train: {train_accuracy * 100:.2f}% | "
                f"Test: {test_accuracy * 100:.2f}%"
            )

    # ------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------

    train_accuracy = calculate_accuracy(
        model,
        train_loader,
        device,
    )

    test_accuracy = calculate_accuracy(
        model,
        test_loader,
        device,
    )

    print()
    print("Results")
    print("-------")

    print(
        f"Training accuracy : "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Test accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # Manual predictions
    # ------------------------------------------------------------

    test_points = torch.tensor(
        [
            [0.8, 0.8],
            [-0.8, -0.8],
            [0.8, -0.8],
            [-0.8, 0.8],

            [0.8, 0.01],
            [0.8, -0.01],
            [0.01, 0.8],
            [-0.01, 0.8],
        ],
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        logits = model(test_points)

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).int()

    print()
    print("Manual predictions")
    print("------------------")

    for point, probability, prediction in zip(
        test_points.cpu(),
        probabilities.cpu(),
        predictions.cpu(),
    ):
        expected_class = int(
            point[0].item()
            * point[1].item()
            > 0
        )

        print(
            f"{point.tolist()} "
            f"expected={expected_class} "
            f"probability={probability.item():.6f} "
            f"predicted={prediction.item()}"
        )


if __name__ == "__main__":
    main()