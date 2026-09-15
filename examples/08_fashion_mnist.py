import torch
import torch.nn as nn

from fashion_mnist import (
    CLASS_NAMES,
    FashionMLP,
    create_dataloaders,
    create_datasets,
    evaluate_epoch,
    train_one_epoch,
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

    print("Image shape :", image.shape)
    print("Image dtype :", image.dtype)
    print(
        "Pixel range :",
        f"{image.min().item():.3f}",
        "->",
        f"{image.max().item():.3f}",
    )
    print(
        "Example label :",
        label,
        "-",
        CLASS_NAMES[label],
    )

    batch_size = 128

    train_loader, test_loader = (
        create_dataloaders(
            train_dataset,
            test_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    )

    print()
    print("DataLoader")
    print("----------")
    print("Batch size :", batch_size)
    print(
        "Training batches :",
        len(train_loader),
    )
    print(
        "Test batches :",
        len(test_loader),
    )

    model = FashionMLP().to(device)

    print()
    print("Model")
    print("-----")
    print(model)

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
        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
            )
        )

        test_loss, test_accuracy = (
            evaluate_epoch(
                model,
                test_loader,
                loss_fn,
                device,
            )
        )

        print(
            f"Epoch {epoch + 1:2d}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test: {test_accuracy * 100:.2f}%"
        )

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

            logits = model(input_image)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_class = int(
                probabilities
                .argmax(dim=1)
                .item()
            )

            confidence = float(
                probabilities[
                    0,
                    predicted_class,
                ].item()
            )

            print(
                f"{index:2d} | "
                f"expected={CLASS_NAMES[label]:12s} | "
                f"predicted={CLASS_NAMES[predicted_class]:12s} | "
                f"confidence={confidence * 100:.2f}%"
            )


if __name__ == "__main__":
    main()
