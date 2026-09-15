from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIRECTORY = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIRECTORY / "binary_classifier.pt"


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


def create_model() -> nn.Module:
    """
    Create the binary classification neural network.
    """

    return nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),

        nn.Linear(16, 16),
        nn.ReLU(),

        nn.Linear(16, 1),
    )


def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> None:
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0,
    )

    epochs = 100

    print()
    print("Training")
    print("--------")

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for inputs, labels in data_loader:
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

            # Calculate loss
            loss = loss_fn(
                logits,
                labels,
            )

            # Reset previous gradients
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            running_loss += (
                loss.item()
                * inputs.size(0)
            )

        epoch_loss = (
            running_loss
            / len(data_loader.dataset)
        )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {epoch_loss:.6f}"
            )


def predict(
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    model.eval()

    with torch.no_grad():
        logits = model(inputs)

        return torch.sigmoid(logits)


def print_predictions(
    points: torch.Tensor,
    probabilities: torch.Tensor,
) -> None:
    for point, probability in zip(
        points.cpu(),
        probabilities.cpu(),
    ):
        expected_class = int(
            point[0].item()
            * point[1].item()
            > 0
        )

        predicted_class = int(
            probability.item() >= 0.5
        )

        print(
            f"{point.tolist()} "
            f"expected={expected_class} "
            f"probability={probability.item():.6f} "
            f"predicted={predicted_class}"
        )


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

    data_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ------------------------------------------------------------
    # Create and train model
    # ------------------------------------------------------------

    model = create_model().to(device)

    print()
    print("Model")
    print("-----")
    print(model)

    train_model(
        model,
        data_loader,
        device,
    )

    # ------------------------------------------------------------
    # Test points
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

    # ------------------------------------------------------------
    # Predictions before saving
    # ------------------------------------------------------------

    probabilities_before = predict(
        model,
        test_points,
    )

    print()
    print("Predictions before saving")
    print("-------------------------")

    print_predictions(
        test_points,
        probabilities_before,
    )

    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print()
    print("Model saved")
    print("-----------")
    print("Path :", MODEL_PATH)

    # ------------------------------------------------------------
    # Create a completely new model
    # ------------------------------------------------------------

    loaded_model = create_model().to(device)

    print()
    print("New model created")
    print("-----------------")

    # At this point, loaded_model contains newly initialized,
    # random weights.

    # ------------------------------------------------------------
    # Load saved parameters
    # ------------------------------------------------------------

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    loaded_model.load_state_dict(
        state_dict
    )

    loaded_model.eval()

    print("Model weights loaded.")

    # ------------------------------------------------------------
    # Predictions after loading
    # ------------------------------------------------------------

    probabilities_after = predict(
        loaded_model,
        test_points,
    )

    print()
    print("Predictions after loading")
    print("-------------------------")

    print_predictions(
        test_points,
        probabilities_after,
    )

    # ------------------------------------------------------------
    # Verify restored model
    # ------------------------------------------------------------

    identical = torch.allclose(
        probabilities_before,
        probabilities_after,
    )

    print()
    print("Verification")
    print("------------")
    print(
        "Predictions are identical :",
        identical,
    )


if __name__ == "__main__":
    main()