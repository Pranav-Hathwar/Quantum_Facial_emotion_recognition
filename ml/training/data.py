"""Torch datasets/loaders on top of the framework-agnostic FER2013 folder dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ml.preprocessing.dataset import FER2013Dataset
from ml.preprocessing.image_preprocessor import preprocess_face


def _build_augmentation(image_size: int):
    """Random training-time augmentation for fine-tuning: applied to the ImageNet-normalised tensor.
    Geometric transforms + random erasing only (no colour jitter — the inputs are already normalised)."""
    from torchvision.transforms import v2 as T  # torchvision transforms operate on CHW tensors

    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=12),
        T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


class FERImageTensorDataset(Dataset):
    """(3,S,S) ImageNet-normalised float tensor + label.
    `hflip`: deterministic horizontal flip of every image (used to build the flipped feature copy).
    `augment`: random train-time augmentation (flip/rotation/crop/erasing) for backbone fine-tuning."""

    def __init__(self, root: Path, split: str, image_size: int = 224, hflip: bool = False,
                 limit: int | None = None, augment: bool = False):
        self.base = FER2013Dataset(root, split)
        if limit is not None:  # deterministic class-balanced-ish subset for smoke tests
            step = max(1, len(self.base.samples) // limit)
            self.base.samples = self.base.samples[::step][:limit]
        self.size, self.hflip = image_size, hflip
        self.aug = _build_augmentation(image_size) if augment else None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        img, label = self.base[i]
        if self.hflip:
            img = np.ascontiguousarray(img[:, ::-1])
        tensor = torch.from_numpy(preprocess_face(img, self.size))
        if self.aug is not None:
            tensor = self.aug(tensor)
        return tensor, label


def feature_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(features.astype(np.float32)), torch.from_numpy(labels.astype(np.int64)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
