"""Evaluate trained models on both RAF-DB and SFEW 2.0 (Zero-Shot Surveillance Transfer).

Computes:
1. Source domain (RAF-DB test set) Accuracy and Macro-F1.
2. Target domain (SFEW 2.0 in-the-wild surveillance) Accuracy and Macro-F1 (Zero-Shot).
3. Domain Transfer Gap = Acc(RAF-DB) - Acc(SFEW).
4. Per-emotion accuracy breakdown on surveillance faces.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ml.config import get_settings
from ml.labels import EMOTIONS, NUM_CLASSES
from ml.model_factory import build_head
from ml.training import feature_cache as fc
from ml.training.data import feature_loader


def evaluate_model_on_cache(ckpt_path: Path, cache_dir: Path, split: str = "test", device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    meta = ckpt.get("meta", {})
    kind = meta.get("kind", "classical")
    hp = meta.get("hyperparameters", {})

    # Build head and load weights
    model = build_head(kind, hp)
    
    # Clean keys if saved with prefixes
    cleaned_state = {}
    for k, v in state.items():
        k_clean = k.replace("head.", "").replace("net.", "")
        cleaned_state[k_clean] = v
    model.load_state_dict(cleaned_state, strict=False)
    model.to(device).eval()

    # Load features
    X, y = fc.load_split(cache_dir, split)
    loader = feature_loader(X, y, batch_size=256, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(by.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    # Per-class accuracy
    per_class = {}
    for idx, emo in enumerate(EMOTIONS):
        mask = all_labels == idx
        if mask.sum() > 0:
            per_class[emo] = float((all_preds[mask] == all_labels[mask]).mean())
        else:
            per_class[emo] = 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_acc": per_class,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(all_labels)
    }


def main():
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--rafdb-cache", type=Path, default=s.model_path / "features_rafdb")
    ap.add_argument("--sfew-cache", type=Path, default=s.model_path / "features_sfew")
    ap.add_argument("--runs-dir", type=Path, default=s.model_path / "runs_rafdb")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    run_dirs = sorted([d for d in a.runs_dir.iterdir() if d.is_dir() and (d / "best.pt").exists()])
    if not run_dirs:
        print(f"[ERROR] No runs found with best.pt in {a.runs_dir}")
        return

    print("=" * 85)
    print(" QUANTUM VISION: RAF-DB vs ZERO-SHOT SURVEILLANCE TRANSFER ON SFEW 2.0")
    print("=" * 85)
    print(f"{'Model Run':<32} | {'RAF-DB Test Acc':<16} | {'SFEW 2.0 Zero-Shot':<18} | {'Transfer Gap':<12}")
    print("-" * 85)

    results = {}
    for r in run_dirs:
        ckpt = r / "best.pt"
        raf_res = evaluate_model_on_cache(ckpt, a.rafdb_cache, split="test", device=a.device)
        sfew_res = evaluate_model_on_cache(ckpt, a.sfew_cache, split="test", device=a.device)
        gap = raf_res["accuracy"] - sfew_res["accuracy"]

        results[r.name] = {
            "raf_db": raf_res,
            "sfew_20": sfew_res,
            "transfer_gap": gap
        }

        print(f"{r.name:<32} | {raf_res['accuracy']*100:6.2f}% (F1:{raf_res['macro_f1']:.3f}) | "
              f"{sfew_res['accuracy']*100:6.2f}% (F1:{sfew_res['macro_f1']:.3f}) | "
              f"{gap*100:6.2f}%")

    print("=" * 85)

    # Save summary json
    summary_path = a.runs_dir / "transfer_evaluation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n[SAVED] Comprehensive report written to: {summary_path}")


if __name__ == "__main__":
    main()
