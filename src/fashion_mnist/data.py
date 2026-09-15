import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from .constants import DATA_DIRECTORY


def create_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(
                torch.float32,
                scale=True,
            ),
        ]
    )


def create_datasets() -> tuple[
    datasets.FashionMNIST,
    datasets.FashionMNIST,
]:
    transform = create_transform()

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

    return train_dataset, test_dataset


def create_dataloaders(
    train_dataset: datasets.FashionMNIST,
    test_dataset: datasets.FashionMNIST,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
