import torch
import torch.nn as nn


def create_dataset(
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a synthetic XOR-like binary classification dataset.

    Class 1: x and y have the same sign.
    Class 0: x and y have different signs.
    """

    inputs = torch.rand(size, 2) * 2 - 1

    labels = (
        inputs[:, 0] * inputs[:, 1] > 0
    ).float().unsqueeze(1)

    return inputs, labels


def calculate_accuracy(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        logits = model(inputs)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).float()

        accuracy = (
            predictions == labels
        ).float().mean()

    return accuracy.item()


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("PyTorch :", torch.__version__)
    print("Device :", device)

    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))
        print("HIP :", torch.version.hip)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    inputs, labels = create_dataset(2000)

    # Mélange des données
    indices = torch.randperm(len(inputs))

    inputs = inputs[indices]
    labels = labels[indices]

    # 80% entraînement / 20% test
    train_size = int(len(inputs) * 0.8)

    x_train = inputs[:train_size].to(device)
    y_train = labels[:train_size].to(device)

    x_test = inputs[train_size:].to(device)
    y_test = labels[train_size:].to(device)

    print()
    print("Dataset")
    print("-------")
    print("Training samples :", len(x_train))
    print("Test samples :", len(x_test))

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
        lr=0.01,
        weight_decay=0.0,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    epochs = 1000

    print()
    print("Training")
    print("--------")

    model.train()

    for epoch in range(epochs):
        # Forward pass
        logits = model(x_train)

        # Calculate loss
        loss = loss_fn(
            logits,
            y_train,
        )

        # Reset previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        if epoch % 100 == 0:
            accuracy = calculate_accuracy(
                model,
                x_train,
                y_train,
            )

            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {loss.item():.6f} | "
                f"Accuracy: {accuracy * 100:.2f}%"
            )

            model.train()

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    train_accuracy = calculate_accuracy(
        model,
        x_train,
        y_train,
    )

    test_accuracy = calculate_accuracy(
        model,
        x_test,
        y_test,
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
    # Misclassified samples
    # ------------------------------------------------------------

    model.eval()

    with torch.no_grad():
        logits = model(x_test)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        incorrect = predictions != y_test

        incorrect_inputs = x_test[incorrect.squeeze()]
        incorrect_labels = y_test[incorrect.squeeze()]
        incorrect_probabilities = probabilities[incorrect.squeeze()]

    print()
    print("Misclassified samples")
    print("---------------------")

    if len(incorrect_inputs) == 0:
        print("None")
    else:
        for point, label, probability in zip(
            incorrect_inputs.cpu(),
            incorrect_labels.cpu(),
            incorrect_probabilities.cpu(),
        ):
            print(
                f"{point.tolist()} "
                f"expected={int(label.item())} "
                f"probability={probability.item():.6f}"
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

            # Points close to the decision boundaries
            [0.8, 0.01],
            [0.8, -0.01],
            [0.01, 0.8],
            [-0.01, 0.8],
        ],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        logits = model(test_points)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).int()

    print()
    print("Manual predictions")
    print("------------------")

    for point, probability, prediction in zip(
        test_points.cpu(),
        probabilities.cpu(),
        predictions.cpu(),
    ):
        expected_class = int(
            point[0].item() * point[1].item() > 0
        )

        print(
            f"{point.tolist()} "
            f"expected={expected_class} "
            f"probability={probability.item():.6f} "
            f"predicted={prediction.item()}"
        )


if __name__ == "__main__":
    main()