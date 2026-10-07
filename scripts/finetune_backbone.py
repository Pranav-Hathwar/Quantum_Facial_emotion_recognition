"""Fine-tune the ResNet50 backbone on FER2013, then re-extract 2048-d features from it.

This lifts the whole pipeline above the frozen-feature ceiling while keeping the fair classical-vs-hybrid
comparison: ONE backbone is fine-tuned (with the classical head, Model A), then frozen and used to
re-cache features; afterwards every head (classical/control/hybrid) is retrained on those features by
run_experiments.py, so differences still come only from the head.

    python finetune_backbone.py --device cuda --epochs 12 --batch-size 48 --unfreeze layer4
    # optional smoke test first:
    python finetune_backbone.py --device cuda --epochs 1 --limit 512 --cache ml/models/features_ft_smoke

Writes: <out>/best.pt (backbone+head), ml/models/backbone_ft.pt (backbone only), and the feature cache.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ml.classical.backbone import ResNet50Backbone
from ml.classical.head import ClassicalResNet50
from ml.config import get_settings
from ml.model_factory import build_head
from ml.training.run import run_extract_features, run_train_finetune
from ml.training.trainer import TrainConfig


def load_finetuned_backbone(best_pt: Path, weights: str, unfreeze: str, device: str) -> ResNet50Backbone:
    """Rebuild the backbone and load the fine-tuned weights out of a ClassicalResNet50 checkpoint."""
    ckpt = torch.load(best_pt, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    backbone = ResNet50Backbone(pretrained=False, mode="fine_tuning", weights=weights, unfreeze=unfreeze)
    bb_state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected backbone keys: {unexpected[:5]}")
    return backbone.to(device)


if __name__ == "__main__":
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)          # low LR for fine-tuning a pretrained backbone
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--unfreeze", choices=["layer4", "all"], default="layer4")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="images/split for a smoke test")
    ap.add_argument("--data", type=Path, default=s.dataset_path)
    ap.add_argument("--out", type=Path, default=s.model_path / "runs_finetune" / "classical_backbone")
    ap.add_argument("--cache", type=Path, default=s.model_path / "features")
    ap.add_argument("--backbone-pt", type=Path, default=s.model_path / "backbone_ft.pt", help="Path to save backbone weights")
    ap.add_argument("--extract-batch", type=int, default=64)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    a = ap.parse_args()

    cfg = TrainConfig(epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, seed=a.seed, device=a.device,
                      amp=not a.no_amp)
    print(f"[fine-tune] unfreeze={a.unfreeze} epochs={a.epochs} bs={a.batch_size} lr={a.lr} seed={a.seed} limit={a.limit}")
    t0 = time.perf_counter()
    hist = run_train_finetune("classical", a.data, a.out, cfg, hp=None, unfreeze=a.unfreeze,
                              num_workers=a.workers, train_limit=a.limit, augment=True)
    print(f"[fine-tune] best val macroF1={hist['best_val_macro_f1']:.4f} "
          f"epoch={hist['best_epoch']} time={ (time.perf_counter()-t0)/60:.1f} min")

    backbone = load_finetuned_backbone(a.out / "best.pt", weights="v1", unfreeze=a.unfreeze, device=a.device)
    a.backbone_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {f"net.{k}": v for k, v in backbone.net.state_dict().items()},
                "meta": {"source": "fine_tuned", "unfreeze": a.unfreeze, "seed": a.seed}},
               a.backbone_pt)
    print(f"[fine-tune] saved backbone -> {a.backbone_pt}")

    if a.skip_extract:
        raise SystemExit(0)
    print(f"[extract] re-extracting features from the fine-tuned backbone -> {a.cache}")
    meta = run_extract_features(a.data, a.cache, image_size=s.image_size, batch_size=a.extract_batch,
                                device=a.device, num_workers=max(1, a.workers - 1), flip_train=True,
                                limit=a.limit, backbone=backbone, backbone_label="resnet50_finetuned_layer4")
    print(f"[extract] done: {meta}")
