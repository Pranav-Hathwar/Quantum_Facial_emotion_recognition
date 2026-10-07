"""Step 1: run the frozen ResNet50 once over FER2013 and cache the 2048-d features.
    python extract_features.py --device cuda
"""
import argparse
from pathlib import Path

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
    ap.add_argument("--limit", type=int, default=None, help="use only N images per split (smoke test)")
    a = ap.parse_args()
    print(run_extract_features(a.data, a.cache, a.image_size, a.batch_size, a.device, a.weights,
                               not a.random_weights, a.workers, not a.no_flip, a.limit))
