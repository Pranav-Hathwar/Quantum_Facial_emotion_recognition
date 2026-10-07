"""Evaluate trained runs on the TEST split and write metrics.json + confusion_matrix.png next to each run.
    python evaluate.py                 # every run under ml/models/runs
    python evaluate.py --run ml/models/runs/hybrid_feature_extractor_q4l2_seed42
"""
import argparse
from pathlib import Path

from ml.config import get_settings
from ml.evaluation.evaluate import evaluate_run

if __name__ == "__main__":
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=s.model_path / "features")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    runs = [a.run] if a.run else sorted(p for p in (s.model_path / "runs").glob("*") if (p / "best.pt").is_file())
    for r in runs:
        m = evaluate_run(r, a.cache, a.split, a.device)
        print(f"{r.name}: acc={m['accuracy']:.4f} macroF1={m['macro']['f1']:.4f} "
              f"train={m['training']['train_time_s']:.0f}s head={m['inference']['head_ms_per_face']:.2f}ms/face")
