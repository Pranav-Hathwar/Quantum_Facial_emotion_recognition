"""High-level entry points: extract features once, then train any head (classical / control / hybrid)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.classical.backbone import ResNet50Backbone, count_parameters
from ml.classical.head import ClassicalResNet50
from ml.model_factory import build_head
from ml.preprocessing.dataset import FER2013Dataset
from ml.training import feature_cache as fc
from ml.training.data import FERImageTensorDataset, feature_loader
from ml.training.trainer import TrainConfig, fit


def run_extract_features(data_root: Path, cache_dir: Path, image_size: int = 224, batch_size: int = 64,
                         device: str = "cpu", weights: str = "v1", pretrained: bool = True, num_workers: int = 2,
                         flip_train: bool = True, limit: int | None = None,
                         backbone: "torch.nn.Module | None" = None, backbone_label: str | None = None) -> dict:
    """Frozen ResNet50 -> 2048-d features for train / validation / test (+ horizontally flipped train copy).
    Pass `backbone` to extract from an already-loaded (e.g. fine-tuned) backbone instead of a fresh ImageNet one."""
    if backbone is None:
        backbone = ResNet50Backbone(pretrained=pretrained, mode="feature_extractor", weights=weights)
    else:  # extracting from a provided backbone: freeze it so BatchNorm stats stay fixed during extraction
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
    meta = {"backbone": backbone_label or "resnet50", "weights": weights if pretrained else "RANDOM (smoke test only)",
            "pretrained": pretrained, "image_size": image_size, "feature_dim": 2048, "device": device,
            "flip_train": flip_train, "limit": limit, "backbone_source": backbone_label or "imagenet"}
    meta["backbone_latency_ms_batch1"] = measure_backbone_latency(backbone, image_size, device)
    for split in ("train", "validation", "test"):
        X, y = fc.extract_features(backbone, data_root, split, image_size, batch_size, device, num_workers, limit=limit)
        fc.save_split(cache_dir, split, X, y, meta)
        if split == "train" and flip_train:
            Xf, yf = fc.extract_features(backbone, data_root, split, image_size, batch_size, device, num_workers,
                                         hflip=True, limit=limit)
            fc.save_split(cache_dir, split, Xf, yf, meta, suffix="_flip")
    return meta


@torch.no_grad()
def measure_backbone_latency(backbone: torch.nn.Module, image_size: int, device: str, repeats: int = 15) -> float:
    """Median milliseconds to run the backbone on ONE face (batch size 1)."""
    backbone.eval().to(device)
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(3):
        backbone(x)
    times = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        backbone(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def run_train_head(kind: str, cache_dir: Path, out_dir: Path, data_root: Path, cfg: TrainConfig,
                   hp: dict | None = None, use_flip: bool = True, log=print) -> dict:
    """Train a head on cached features (feature_extractor mode)."""
    hp = dict(hp or {})
    X, y = fc.load_split(cache_dir, "train", include_flip=use_flip)
    Xv, yv = fc.load_split(cache_dir, "validation")
    train_loader = feature_loader(X, y, cfg.batch_size, shuffle=True)
    val_loader = feature_loader(Xv, yv, 512, shuffle=False)
    weights = FER2013Dataset(data_root, "train").class_weights()
    model = build_head(kind, hp)
    import json as _json
    cache_meta = _json.loads((Path(cache_dir) / "meta.json").read_text())
    meta = {"kind": kind, "mode": "feature_extractor", "hyperparameters": hp, "seed": cfg.seed,
            "head_parameters": count_parameters(model), "feature_cache": str(cache_dir),
            "backbone_pretrained": cache_meta.get("pretrained"), "backbone_weights": cache_meta.get("weights"),
            "image_size": cache_meta.get("image_size")}
    log(f"[{kind}] seed={cfg.seed}  head parameters={meta['head_parameters']:,}  train samples={len(y):,}")
    return fit(model, train_loader, val_loader, cfg, out_dir, weights, meta, log)


def run_train_finetune(kind: str, data_root: Path, out_dir: Path, cfg: TrainConfig, hp: dict | None = None,
                       image_size: int = 224, weights: str = "v1", unfreeze: str = "layer4",
                       pretrained: bool = True, num_workers: int = 4, train_limit: int | None = None,
                       augment: bool = True, log=print) -> dict:
    """End-to-end fine-tuning (RESNET_MODE=fine_tuning): backbone layer4 + head trained on raw images. Needs a GPU for real use."""
    hp = dict(hp or {})
    train_ds = FERImageTensorDataset(data_root, "train", image_size, hflip=False, limit=train_limit, augment=augment)
    val_ds = FERImageTensorDataset(data_root, "validation", image_size, limit=train_limit)
    loader_kw = dict(num_workers=num_workers, pin_memory=str(cfg.device).startswith("cuda"))
    if num_workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, 128, shuffle=False, **loader_kw)
    backbone = ResNet50Backbone(pretrained=pretrained, mode="fine_tuning", weights=weights, unfreeze=unfreeze)
    model = ClassicalResNet50(backbone, build_head(kind, hp))
    meta = {"kind": kind, "mode": "fine_tuning", "hyperparameters": hp, "seed": cfg.seed, "unfreeze": unfreeze,
            "weights": weights, "image_size": image_size, "backbone_pretrained": pretrained, "augment": augment,
            "head_parameters": count_parameters(model.head), "trainable_parameters": count_parameters(model, True)}
    log(f"[{kind}/fine_tuning] trainable parameters={meta['trainable_parameters']:,}")
    return fit(model, train_loader, val_loader, cfg, out_dir, FER2013Dataset(data_root, "train").class_weights(), meta, log)
