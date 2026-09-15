from pathlib import Path

import torch
import torch.nn as nn

from fashion_mnist import (
    FashionCNN,
    create_datasets,
    create_train_validation_test_dataloaders,
    evaluate_epoch,
    load_training_checkpoint,
    save_training_checkpoint,
    train_one_epoch,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "fashion_cnn_best.pt"
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

    train_dataset, test_dataset = (
        create_datasets()
    )

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_train_validation_test_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=128,
        validation_ratio=0.1,
        seed=42,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print("Dataset")
    print("-------")

    print(
        "Training samples :",
        len(train_loader.dataset),
    )

    print(
        "Validation samples :",
        len(validation_loader.dataset),
    )

    print(
        "Test samples :",
        len(test_loader.dataset),
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
    # Training configuration
    # ------------------------------------------------------------

    max_epochs = 30

    patience = 4

    min_delta = 1e-4

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    print()
    print("Training")
    print("--------")

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        (
            validation_loss,
            validation_accuracy,
        ) = evaluate_epoch(
            model,
            validation_loader,
            loss_fn,
            device,
        )

        improved = (
            validation_loss
            < best_validation_loss
            - min_delta
        )

        marker = ""

        if improved:
            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            save_training_checkpoint(
                CHECKPOINT_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                validation_loss=(
                    validation_loss
                ),
            )

            marker = " *"

        else:
            epochs_without_improvement += 1

        print(
            f"Epoch "
            f"{epoch:2d}/{max_epochs} | "
            f"Train loss: "
            f"{train_loss:.4f} | "
            f"Train: "
            f"{train_accuracy * 100:.2f}% | "
            f"Val loss: "
            f"{validation_loss:.4f} | "
            f"Val: "
            f"{validation_accuracy * 100:.2f}%"
            f"{marker}"
        )

        if (
            epochs_without_improvement
            >= patience
        ):
            print()
            print(
                "Early stopping triggered."
            )

            break

    # ------------------------------------------------------------
    # Restore best checkpoint
    # ------------------------------------------------------------

    (
        best_epoch,
        best_validation_loss,
    ) = load_training_checkpoint(
        CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    print()
    print("Best checkpoint")
    print("---------------")

    print(
        "Epoch :",
        best_epoch,
    )

    print(
        "Validation loss :",
        f"{best_validation_loss:.4f}",
    )

    print(
        "Path :",
        CHECKPOINT_PATH,
    )

    # ------------------------------------------------------------
    # Final test
    # ------------------------------------------------------------

    test_loss, test_accuracy = (
        evaluate_epoch(
            model,
            test_loader,
            loss_fn,
            device,
        )
    )

    print()
    print("Final test")
    print("----------")

    print(
        "Test loss :",
        f"{test_loss:.4f}",
    )

    print(
        "Test accuracy :",
        f"{test_accuracy * 100:.2f}%",
    )


if __name__ == "__main__":
    main()