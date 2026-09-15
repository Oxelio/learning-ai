from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision.transforms import v2

from .constants import CLASS_NAMES


def create_inference_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((28, 28), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])


def load_external_image(path: Path, *, invert: bool = False) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if invert:
        image = ImageOps.invert(image)
    return create_inference_transform()(image)


def predict_image(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> tuple[int, float, torch.Tensor]:
    model.eval()
    input_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]
    predicted_class = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_class].item())
    return predicted_class, confidence, probabilities.cpu()


def format_prediction(predicted_class: int, confidence: float) -> str:
    return f"{CLASS_NAMES[predicted_class]} ({confidence * 100:.2f}%)"
