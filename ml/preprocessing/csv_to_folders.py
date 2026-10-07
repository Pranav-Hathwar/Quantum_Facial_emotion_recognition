"""Convert the Kaggle fer2013.csv into the folder layout used by this project.

    data/fer2013/{train,validation,test}/{angry,disgust,fear,happy,sad,surprise,neutral}/*.png

Split mapping (the dataset's own 'Usage' column):
    Training -> train, PublicTest -> validation, PrivateTest -> test
If the CSV has no 'Usage' column, a seeded 80/10/10 split is used.

Usage:
    python -m ml.preprocessing.csv_to_folders --csv path/to/fer2013.csv [--out data/fer2013] [--overwrite]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ml.config import PROJECT_ROOT
from ml.labels import EMOTIONS, NUM_CLASSES

IMG_SIDE = 48
USAGE_TO_SPLIT = {"training": "train", "publictest": "validation", "privatetest": "test"}
SPLITS = ("train", "validation", "test")


def _parse_pixels(text: str) -> np.ndarray | None:
    values = np.array(text.split(), dtype=np.int64)
    if values.size != IMG_SIDE * IMG_SIDE or values.min() < 0 or values.max() > 255:
        return None
    return values.astype(np.uint8).reshape(IMG_SIDE, IMG_SIDE)


def _fallback_split(rng: random.Random) -> str:
    r = rng.random()
    return "train" if r < 0.8 else ("validation" if r < 0.9 else "test")


def convert(csv_path: Path, out_dir: Path, overwrite: bool = False, limit: int | None = None, seed: int = 42) -> dict:
    csv_path, out_dir = Path(csv_path), Path(out_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    for split in SPLITS:
        for emo in EMOTIONS:
            (out_dir / split / emo).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    skipped = Counter()
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = {(h or "").strip().lower(): h for h in (reader.fieldnames or [])}
        if "emotion" not in header or "pixels" not in header:
            raise ValueError(f"CSV must have 'emotion' and 'pixels' columns, found {reader.fieldnames}")
        usage_col = header.get("usage")

        for row_idx, row in enumerate(reader):
            if limit is not None and row_idx >= limit:
                break
            try:
                label = int(row[header["emotion"]])
            except (TypeError, ValueError):
                skipped["bad_label"] += 1
                continue
            if not 0 <= label < NUM_CLASSES:
                skipped["bad_label"] += 1
                continue
            img = _parse_pixels(row[header["pixels"]])
            if img is None:
                skipped["bad_pixels"] += 1
                continue
            if usage_col:
                split = USAGE_TO_SPLIT.get((row[usage_col] or "").strip().lower())
                if split is None:
                    skipped["bad_usage"] += 1
                    continue
            else:
                split = _fallback_split(rng)

            emo = EMOTIONS[label]
            target = out_dir / split / emo / f"{row_idx:06d}.png"
            if target.exists() and not overwrite:
                skipped["already_exists"] += 1
                counts[split][emo] += 1
                continue
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                skipped["encode_error"] += 1
                continue
            target.write_bytes(buf.tobytes())  # write_bytes avoids Windows non-ASCII path issues in cv2.imwrite
            counts[split][emo] += 1

    summary = {
        "source_csv": str(csv_path),
        "output_dir": str(out_dir),
        "counts": {s: {e: counts[s][e] for e in EMOTIONS} for s in SPLITS},
        "totals": {s: sum(counts[s].values()) for s in SPLITS},
        "skipped": dict(skipped),
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path, help="path to fer2013.csv")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "fer2013")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="only convert the first N rows (for quick tests)")
    args = p.parse_args(argv)
    summary = convert(args.csv, args.out, overwrite=args.overwrite, limit=args.limit)
    print(json.dumps({"totals": summary["totals"], "skipped": summary["skipped"]}, indent=2))
    print(f"Done. Per-class counts written to {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
