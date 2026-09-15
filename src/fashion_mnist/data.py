import torch

from torch.utils.data import (
    DataLoader,
    random_split,
)

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


def create_train_validation_test_dataloaders(
    train_dataset: datasets.FashionMNIST,
    test_dataset: datasets.FashionMNIST,
    *,
    batch_size: int = 128,
    validation_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    validation_size = int(
        len(train_dataset)
        * validation_ratio
    )

    training_size = (
        len(train_dataset)
        - validation_size
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    training_dataset, validation_dataset = (
        random_split(
            train_dataset,
            [
                training_size,
                validation_size,
            ],
            generator=generator,
        )
    )

    train_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    return (
        train_loader,
        validation_loader,
        test_loader,
    )