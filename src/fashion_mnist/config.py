from __future__ import annotations

from dataclasses import dataclass
from json import load
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 128
    validation_ratio: float = 0.1
    max_epochs: int = 30
    patience: int = 4
    min_delta: float = 1e-4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0

    @classmethod
    def from_json(cls, path: Path) -> "TrainingConfig":
        with path.open("r", encoding="utf-8") as file:
            values = load(file)
        return cls(**values)
