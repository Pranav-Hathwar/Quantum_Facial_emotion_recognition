"""Cache frozen-ResNet50 features once, so every head (classical / control / hybrid) trains on identical inputs.

Only valid when the backbone is frozen (feature_extractor mode). Features are stored as float16 to save disk.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.training.data import FERImageTensorDataset


@torch.no_grad()
def extract_features(backbone: torch.nn.Module, root: Path, split: str, image_size: int, batch_size: int,
                     device: str, num_workers: int = 2, hflip: bool = False, limit: int | None = None,
                     progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
    ds = FERImageTensorDataset(root, split, image_size, hflip=hflip, limit=limit)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    backbone.eval().to(device)
    feats, labels = [], []
    t0 = time.time()
    tag = f"{split}{' (hflip)' if hflip else ''}"
    print(f"\n[INFO] Starting extraction for split '{tag}': {len(ds):,} images total", flush=True)

    for i, (x, y) in enumerate(loader):
        feats.append(backbone(x.to(device, non_blocking=True)).float().cpu().numpy().astype(np.float16))
        labels.append(y.numpy())
        if progress and (i % 25 == 0 or i == len(loader) - 1):
            done = min((i + 1) * batch_size, len(ds))
            elapsed = time.time() - t0
            speed = done / max(elapsed, 0.001)
            pct = 100.0 * done / len(ds)
            print(f"  -> [{tag}] {done:,}/{len(ds):,} images ({pct:5.1f}%) | {speed:.1f} img/s | {elapsed:.0f}s elapsed", flush=True)

    elapsed_total = time.time() - t0
    print(f"[DONE] Completed '{tag}' in {elapsed_total:.1f}s ({len(ds) / max(elapsed_total, 0.001):.1f} img/s avg)\n", flush=True)
    return np.concatenate(feats), np.concatenate(labels)


def save_split(cache_dir: Path, split: str, feats: np.ndarray, labels: np.ndarray, meta: dict, suffix: str = "") -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}{suffix}.npz"
    np.savez(path, features=feats, labels=labels)
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[SAVED] Feature file written: {path} ({size_mb:.1f} MB)", flush=True)
    return path


def load_split(cache_dir: Path, split: str, include_flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = Path(cache_dir)
    p = cache_dir / f"{split}.npz"
    if not p.is_file():
        raise FileNotFoundError(f"No cached features at {p}. Run: python extract_features.py")
    d = np.load(p)
    X, y = d["features"], d["labels"]
    flip = cache_dir / f"{split}_flip.npz"
    if include_flip and flip.is_file():
        f = np.load(flip)
        X, y = np.concatenate([X, f["features"]]), np.concatenate([y, f["labels"]])
    return X, y
