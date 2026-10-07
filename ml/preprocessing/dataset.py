"""FER2013 folder dataset (framework-agnostic: works with torch DataLoader as a map-style dataset)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ml.exceptions import DatasetNotFoundError
from ml.labels import EMOTIONS, NUM_CLASSES

VALID_SPLITS = ("train", "validation", "test")
_IMG_EXT = {".png", ".jpg", ".jpeg"}


def list_samples(root: Path, split: str) -> list[tuple[Path, int]]:
    """Return [(image_path, class_index)] for data/fer2013/<split>/<emotion>/*.png (sorted, deterministic)."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise DatasetNotFoundError(
            f"FER2013 split folder not found: {split_dir}\n"
            "Place fer2013.csv somewhere and run:\n"
            "  python -m ml.preprocessing.csv_to_folders --csv path/to/fer2013.csv"
        )
    samples: list[tuple[Path, int]] = []
    for idx, emo in enumerate(EMOTIONS):
        for p in sorted((split_dir / emo).glob("*")):
            if p.suffix.lower() in _IMG_EXT:
                samples.append((p, idx))
    if not samples:
        raise DatasetNotFoundError(f"No images found under {split_dir}. Did the CSV conversion finish?")
    return samples


class FER2013Dataset:
    """Returns (image, label). image is uint8 HxW grayscale (or HxWx3 BGR if as_rgb=True), unless `transform` changes it."""

    def __init__(self, root: Path, split: str, transform: Callable | None = None, as_rgb: bool = False):
        self.root, self.split, self.transform, self.as_rgb = Path(root), split, transform, as_rgb
        self.samples = list_samples(self.root, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not decode image: {path}")
        if self.as_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> dict[str, int]:
        c = Counter(label for _, label in self.samples)
        return {EMOTIONS[i]: c.get(i, 0) for i in range(NUM_CLASSES)}

    def class_weights(self) -> list[float]:
        """Inverse-frequency weights (mean-normalised) for the imbalanced FER2013 classes (Disgust is rare)."""
        counts = np.array([max(v, 1) for v in self.class_counts().values()], dtype=np.float64)
        w = counts.sum() / (NUM_CLASSES * counts)
        return (w / w.mean()).tolist()
