from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from fashion_mnist import (
    FashionCNN, TrainingConfig, create_datasets,
    create_train_validation_test_dataloaders, evaluate_epoch,
    save_training_checkpoint, train_one_epoch,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "fashion_mnist.json"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "fashion_cnn_best.pt"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "outputs" / "training_history.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Fashion-MNIST CNN.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_json(args.config)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch :", torch.__version__)
    print("Device :", device)
    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))
        print("HIP :", torch.version.hip)

    train_dataset, test_dataset = create_datasets()
    train_loader, validation_loader, _ = create_train_validation_test_dataloaders(
        train_dataset, test_dataset,
        batch_size=config.batch_size,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nDataset\n-------")
    print("Training samples :", len(train_loader.dataset))
    print("Validation samples :", len(validation_loader.dataset))

    model = FashionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []

    print("\nTraining\n--------")
    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        validation_loss, validation_accuracy = evaluate_epoch(
            model, validation_loader, loss_fn, device
        )
        improved = validation_loss < best_validation_loss - config.min_delta
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            save_training_checkpoint(
                args.checkpoint,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                validation_loss=validation_loss,
                config=asdict(config),
            )
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "checkpoint_saved": improved,
        })
        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:2d}/{config.max_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train: {train_accuracy * 100:.2f}% | "
            f"Val loss: {validation_loss:.4f} | "
            f"Val: {validation_accuracy * 100:.2f}%{marker}"
        )
        if epochs_without_improvement >= config.patience:
            print("\nEarly stopping triggered.")
            break

    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("w", encoding="utf-8") as file:
        json.dump({"config": asdict(config), "history": history}, file, indent=2)

    print("\nArtifacts\n---------")
    print("Checkpoint :", args.checkpoint)
    print("History    :", args.history)


if __name__ == "__main__":
    main()
