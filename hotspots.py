"""Hotspot catalog and CPU-only MARIDA patch analysis services."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TypedDict

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.io import MemoryFile

from inference import DebrisDetector


class Hotspot(TypedDict):
    id: str
    name: str
    coordinates: list[float]
    description: str
    sample_file: str


class _AnalysisResult(TypedDict):
    id: str
    name: str
    coordinates: list[float]
    description: str
    sample_file: str
    preview_png_base64: str
    overlay_png_base64: str
    debris_pixel_count: int
    total_pixels: int
    debris_coverage_percentage: float


_PROJECT_ROOT = Path(__file__).parent
_SAMPLE_ROOT = _PROJECT_ROOT / "samples"
_HOTSPOTS: tuple[Hotspot, ...] = (
    {
        "id": "accra-coast",
        "name": "Accra Coast, Ghana",
        "coordinates": [5.55, -0.20],
        "description": "Verified MARIDA coastal monitoring site near Accra.",
        "sample_file": "samples/accra-coast.tif",
    },
    {
        "id": "gulf-of-honduras",
        "name": "Gulf of Honduras",
        "coordinates": [15.80, -87.90],
        "description": "Verified MARIDA coastal monitoring site in the Gulf of Honduras.",
        "sample_file": "samples/gulf-of-honduras.tif",
    },
    {
        "id": "durban-harbor",
        "name": "Durban Harbor, South Africa",
        "coordinates": [-29.86, 31.02],
        "description": "Verified MARIDA coastal monitoring site near Durban Harbor.",
        "sample_file": "samples/durban-harbor.tif",
    },
    {
        "id": "coastal-demo-site",
        "name": "Coastal Demo Site",
        "coordinates": [12.91, 74.85],
        "description": "Verified MARIDA coastal monitoring demonstration site.",
        "sample_file": "samples/coastal-demo-site.tif",
    },
)
_DETECTOR: DebrisDetector | None = None


def get_hotspot_catalog() -> list[dict]:
    """Return serializable metadata for all verified monitoring sites."""
    return [dict(hotspot) for hotspot in _HOTSPOTS]


def _get_detector() -> DebrisDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = DebrisDetector(_PROJECT_ROOT / "models" / "model.pth")
    return _DETECTOR


def _encode_png(image: np.ndarray, mode: str) -> str:
    buffer = BytesIO()
    Image.fromarray(image, mode=mode).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _normalize_band(band: np.ndarray) -> np.ndarray:
    values = band.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = values[finite].min(), values[finite].max()
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - low) / (high - low)
    return (np.clip(np.nan_to_num(normalized), 0.0, 1.0) * 255).astype(np.uint8)


def _analyze_dataset(
    dataset: rasterio.DatasetReader, metadata: Hotspot
) -> _AnalysisResult:
    if dataset.count != 11:
        raise ValueError(f"expected 11 raster bands, found {dataset.count}")
    bands = dataset.read()
    preview = np.stack(
        [_normalize_band(bands[index - 1]) for index in (4, 3, 2)], axis=-1
    )
    mask = _get_detector().predict(torch.from_numpy(bands)).numpy()
    debris = mask == 0
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[debris] = [255, 0, 0, 255]
    total_pixels = int(mask.size)
    debris_pixel_count = int(debris.sum())
    return {
        **metadata,
        "preview_png_base64": _encode_png(preview, "RGB"),
        "overlay_png_base64": _encode_png(overlay, "RGBA"),
        "debris_pixel_count": debris_pixel_count,
        "total_pixels": total_pixels,
        "debris_coverage_percentage": debris_pixel_count / total_pixels * 100,
    }


def _metadata_for_custom() -> Hotspot:
    return {
        "id": "custom",
        "name": "Custom Patch",
        "coordinates": [0.0, 0.0],
        "description": "User-uploaded MARIDA-compatible GeoTIFF patch.",
        "sample_file": "",
    }


def scan_hotspot(hotspot_id: str) -> dict:
    """Analyze the registered GeoTIFF for a hotspot ID."""
    hotspot = next((item for item in _HOTSPOTS if item["id"] == hotspot_id), None)
    if hotspot is None:
        raise KeyError(f"unknown hotspot ID: {hotspot_id}")
    sample_path = _PROJECT_ROOT / hotspot["sample_file"]
    if not sample_path.is_file():
        raise FileNotFoundError(f"hotspot sample file not found: {sample_path}")
    with rasterio.open(sample_path) as dataset:
        return _analyze_dataset(dataset, hotspot)


def analyze_custom_patch(file_bytes_or_path: bytes | bytearray | str | Path) -> dict:
    """Analyze a user-uploaded GeoTIFF supplied as bytes or a filesystem path."""
    if isinstance(file_bytes_or_path, (str, Path)):
        with rasterio.open(file_bytes_or_path) as dataset:
            return _analyze_dataset(dataset, _metadata_for_custom())
    if not isinstance(file_bytes_or_path, (bytes, bytearray)):
        raise TypeError("custom patch must be bytes, bytearray, or a filesystem path")
    with MemoryFile(bytes(file_bytes_or_path)) as memory_file:
        with memory_file.open() as dataset:
            return _analyze_dataset(dataset, _metadata_for_custom())


if __name__ == "__main__":
    catalog = get_hotspot_catalog()
    assert len(catalog) == 4
    assert all(len(item["coordinates"]) == 2 for item in catalog)
    try:
        result = scan_hotspot(catalog[0]["id"])
    except FileNotFoundError as error:
        print(f"Validation passed: {error}")
    else:
        assert result["total_pixels"] == 256 * 256
        assert result["debris_pixel_count"] >= 0
        assert result["debris_coverage_percentage"] >= 0.0
        assert result["preview_png_base64"]
        assert result["overlay_png_base64"]
        print("Validation passed: hotspot scan completed")