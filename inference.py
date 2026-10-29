"""CPU-only inference for the MARIDA U-Net segmentation model."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch import Tensor

from models.unet import UNet


class DebrisDetector:
    def __init__(
        self,
        model_path: str | Path = "models/model.pth",
        input_bands: int = 11,
        output_classes: int = 11,
        hidden_channels: int = 16,
    ) -> None:
        self.device = torch.device("cpu")
        self.input_bands = input_bands
        self.model = UNet(
            input_bands=input_bands,
            output_classes=output_classes,
            hidden_channels=hidden_channels,
        ).to(self.device)
        checkpoint = torch.load(
            Path(model_path),
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(self._state_dict_from_checkpoint(checkpoint))
        self.model.eval()

    @staticmethod
    def _state_dict_from_checkpoint(checkpoint: Any) -> Mapping[str, Tensor]:
        if isinstance(checkpoint, Mapping) and all(
            isinstance(key, str) for key in checkpoint
        ):
            for key in ("state_dict", "model_state_dict"):
                nested = checkpoint.get(key)
                if isinstance(nested, Mapping):
                    checkpoint = nested
                    break
        if not isinstance(checkpoint, Mapping) or not checkpoint:
            raise TypeError("model checkpoint must contain a non-empty state dictionary")
        if not all(isinstance(key, str) and isinstance(value, Tensor)
                   for key, value in checkpoint.items()):
            raise TypeError("model checkpoint contains invalid state dictionary values")
        return checkpoint

    @torch.inference_mode()
    def predict(self, image: Tensor) -> Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != self.input_bands:
            raise ValueError(
                f"image must have shape ({self.input_bands}, H, W) or "
                f"(1, {self.input_bands}, H, W)"
            )
        logits = self.model(image.to(device=self.device, dtype=torch.float32))
        return logits.argmax(dim=1).squeeze(0)

    __call__ = predict


def run_inference_on_geotiff(image_path: str | Path) -> np.ndarray:
    """Run CPU inference on an 11-band GeoTIFF and return its 2D class mask."""
    with rasterio.open(image_path) as source:
        if source.count != 11:
            raise ValueError(f"expected 11 raster bands, found {source.count}")
        image = torch.from_numpy(source.read())

    return DebrisDetector().predict(image).numpy()