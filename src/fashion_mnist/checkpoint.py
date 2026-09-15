from pathlib import Path

import torch
import torch.nn as nn


def save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    validation_loss: float,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "validation_loss":
                validation_loss,
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    epoch = int(
        checkpoint["epoch"]
    )

    validation_loss = float(
        checkpoint["validation_loss"]
    )

    return (
        epoch,
        validation_loss,
    )