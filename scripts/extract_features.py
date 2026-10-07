"""Step 1: run the frozen ResNet50 once over FER2013 and cache the 2048-d features.
    python extract_features.py --device cuda
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import get_settings
from ml.training.run import run_extract_features

if __name__ == "__main__":
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=s.dataset_path)
    ap.add_argument("--cache", type=Path, default=s.model_path / "features")
    ap.add_argument("--device", default=s.device)
    ap.add_argument("--image-size", type=int, default=s.image_size)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weights", choices=["v1", "v2"], default="v1")
    ap.add_argument("--no-flip", action="store_true", help="skip the flipped training copy")
    ap.add_argument("--random-weights", action="store_true", help="SMOKE TEST ONLY: no pretrained weights")
    ap.add_argument("--backbone-pt", type=Path, default=None, help="Path to custom backbone weights (e.g. fine-tuned on AffectNet)")
    ap.add_argument("--backbone-label", type=str, default=None, help="Label for feature cache metadata")
    ap.add_argument("--limit", type=int, default=None, help="use only N images per split (smoke test)")
    a = ap.parse_args()

    print("=" * 70, flush=True)
    print(" QuantumVision: ResNet-50 Feature Extraction", flush=True)
    print("=" * 70, flush=True)
    print(f"  - Device:          {a.device}", flush=True)
    print(f"  - Batch size:      {a.batch_size}", flush=True)
    print(f"  - Image size:      {a.image_size}x{a.image_size}", flush=True)
    print(f"  - Pretrained:      {not a.random_weights} ({a.weights})", flush=True)
    print(f"  - Custom Backbone: {a.backbone_pt}", flush=True)
    print(f"  - Flip copy:       {not a.no_flip}", flush=True)
    print(f"  - Dataset path:    {a.data}", flush=True)
    print(f"  - Cache path:      {a.cache}", flush=True)
    print("=" * 70, flush=True)

    backbone = None
    backbone_label = a.backbone_label or "resnet50_imagenet"
    if a.backbone_pt and a.backbone_pt.exists():
        import torch
        from ml.classical.backbone import ResNet50Backbone
        print(f"  [INFO] Loading custom backbone weights from {a.backbone_pt}...", flush=True)
        backbone = ResNet50Backbone(pretrained=False, mode="feature_extractor", weights=a.weights)
        ckpt = torch.load(a.backbone_pt, map_location=a.device)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        state = {k.replace("net.", ""): v for k, v in state.items()}
        backbone.net.load_state_dict(state, strict=False)
        backbone = backbone.to(a.device)
        backbone_label = a.backbone_label or "resnet50_affectnet"

    meta = run_extract_features(a.data, a.cache, a.image_size, a.batch_size, a.device, a.weights,
                                not a.random_weights, a.workers, not a.no_flip, a.limit,
                                backbone=backbone, backbone_label=backbone_label)

    print("=" * 70, flush=True)
    print(" [SUCCESS] Feature extraction completed successfully!", flush=True)
    print(f" Cached files are saved in: {a.cache}", flush=True)
    print(f" Backbone latency: {meta.get('backbone_latency_ms_batch1', 0):.2f} ms/face", flush=True)
    print("=" * 70, flush=True)
