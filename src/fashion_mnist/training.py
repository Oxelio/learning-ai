import torch
import torch.nn as nn

from torch.utils.data import DataLoader


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
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total += batch_size

    return (
        total_loss / total,
        correct / total,
    )


def evaluate_epoch(
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
            loss = loss_fn(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total += batch_size

    return (
        total_loss / total,
        correct / total,
    )
