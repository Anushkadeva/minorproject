"""Synthetic CPU smoke test for the MARIDA inference pipeline."""

from pathlib import Path

import torch

from inference import DebrisDetector


def main() -> None:
    detector = DebrisDetector(Path(__file__).parent / "models" / "model.pth")
    image = torch.randn(1, 11, 64, 64, device=torch.device("cpu"))

    batch_mask = detector.predict(image)
    unbatched_mask = detector.predict(image.squeeze(0))

    for mask in (batch_mask, unbatched_mask):
        assert mask.shape == (64, 64), mask.shape
        assert mask.device == torch.device("cpu"), mask.device
        assert mask.dtype == torch.int64, mask.dtype
        assert int(mask.min()) >= 0
        assert int(mask.max()) < 11

    assert torch.equal(batch_mask, unbatched_mask)
    print("Synthetic CPU inference test passed:", tuple(batch_mask.shape))


if __name__ == "__main__":
    main()