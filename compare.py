"""Build the classical-vs-hybrid comparison from evaluated runs:  python compare.py"""
import argparse
from pathlib import Path

from ml.config import get_settings
from ml.evaluation.compare import build_comparison

if __name__ == "__main__":
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=s.model_path / "runs")
    a = ap.parse_args()
    res = build_comparison(a.runs)
    print(res["summary"])
    print(f"Wrote {a.runs.parent / 'comparison.md'}")
