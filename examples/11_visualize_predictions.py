import torch
import torch.nn as nn

from fashion_mnist import (
    CLASS_NAMES,
    OUTPUT_DIRECTORY,
    FashionCNN,
    create_dataloaders,
    create_datasets,
    evaluate_classifier,
    train_one_epoch,
)
from fashion_mnist.visualization import (
    save_error_grid,
)


OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "fashion_mnist_errors.png"
)


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

    train_dataset, test_dataset = (
        create_datasets()
    )

    train_loader, test_loader = (
        create_dataloaders(
            train_dataset,
            test_dataset,
            batch_size=128,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    )

    model = FashionCNN().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0,
    )

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

    result = evaluate_classifier(
        model,
        test_loader,
        device,
        collect_images=True,
    )

    print()
    print("Errors")
    print("------")
    print(
        "Misclassified samples :",
        len(result.errors),
    )

    errors = sorted(
        result.errors,
        key=lambda error: error.confidence,
        reverse=True,
    )

    print()
    print("Most confident errors")
    print("---------------------")

    for error in errors[:10]:
        print(
            f"expected="
            f"{CLASS_NAMES[error.expected]:12s} | "
            f"predicted="
            f"{CLASS_NAMES[error.predicted]:12s} | "
            f"predicted confidence="
            f"{error.confidence * 100:6.2f}% | "
            f"expected probability="
            f"{error.expected_probability * 100:6.2f}%"
        )

    save_error_grid(
        errors,
        OUTPUT_PATH,
        count=16,
        columns=4,
        show=True,
    )

    print()
    print(
        "Visualization saved :",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
