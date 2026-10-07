"""Evaluate a trained checkpoint on the held-out test split and write metrics.json (+ confusion matrix PNG).

Every number in the dashboard is read from these files. Nothing is hand-entered.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ml.classical.backbone import count_parameters
from ml.evaluation.metrics import compute_metrics
from ml.labels import EMOTIONS
from ml.model_factory import load_head
from ml.training import feature_cache as fc
from ml.training.data import feature_loader
from ml.training.trainer import predict_probs


@torch.no_grad()
def head_latency_ms(model: torch.nn.Module, device: str, in_features: int = 2048, repeats: int = 100) -> float:
    """Median milliseconds for ONE face through the head (for the hybrid model this includes the VQC simulation)."""
    model.eval().to(device)
    x = torch.randn(1, in_features, device=device)
    for _ in range(5):
        model(x)
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def save_confusion_png(cm: list[list[int]], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), [e.capitalize() for e in EMOTIONS], rotation=45, ha="right")
    ax.set_yticks(range(7), [e.capitalize() for e in EMOTIONS])
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def evaluate_run(run_dir: Path, cache_dir: Path, split: str = "test", device: str = "cpu") -> dict:
    run_dir = Path(run_dir)
    model, meta = load_head(run_dir / "best.pt", device)
    X, y = fc.load_split(cache_dir, split)
    probs, labels = predict_probs(model, feature_loader(X, y, 512, shuffle=False), device)
    result = compute_metrics(labels, probs.argmax(1))
    np.savez_compressed(run_dir / f"{split}_predictions.npz", probs=probs.astype(np.float32), labels=labels)
    history = json.loads((run_dir / "history.json").read_text()) if (run_dir / "history.json").is_file() else {}
    cache_meta = json.loads((Path(cache_dir) / "meta.json").read_text())
    backbone_ms = cache_meta.get("backbone_latency_ms_batch1")
    head_ms = head_latency_ms(model, device)
    result.update({
        "model_kind": meta["kind"], "split": split, "seed": meta.get("seed"),
        "hyperparameters": meta.get("hyperparameters", {}),
        "parameters": {"head_total": count_parameters(model),
                       "quantum_circuit": int(model.quantum.weights.numel()) if hasattr(model, "quantum") else 0},
        "training": {"train_time_s": history.get("train_time_s"), "epochs_run": history.get("epochs_run"),
                     "best_epoch": history.get("best_epoch"), "best_val_macro_f1": history.get("best_val_macro_f1")},
        "inference": {"head_ms_per_face": head_ms, "backbone_ms_per_face": backbone_ms,
                      "total_ms_per_face": (head_ms + backbone_ms) if backbone_ms is not None else None,
                      "device": device, "note": "median of single-face calls; hybrid head time includes classical simulation of the VQC"},
        "backbone": {k: cache_meta.get(k) for k in ("weights", "pretrained", "image_size")},
    })
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    save_confusion_png(result["confusion_matrix"], run_dir / "confusion_matrix.png",
                       f"{meta['kind']} (seed {meta.get('seed')}) - {split}")
    return result
