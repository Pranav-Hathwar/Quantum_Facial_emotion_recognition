"""Augment underrepresented training classes in FER2013 to achieve balanced class sizes.

Only the 'train' split is augmented; 'validation' and 'test' splits remain untouched.

Usage:
    py -3.12 scripts/augment_dataset.py [--target 7215]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import PROJECT_ROOT
from ml.labels import EMOTIONS


def augment_image(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply realistic facial emotion preserving augmentations."""
    h, w = img.shape[:2]

    # 1. Random horizontal flip (p=0.5)
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)

    # 2. Random rotation (-12 to +12 degrees) and slight scale (0.95 to 1.05)
    angle = rng.uniform(-12.0, 12.0)
    scale = rng.uniform(0.95, 1.05)
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 3. Random small translation (-2 to +2 pixels)
    matrix[0, 2] += rng.uniform(-2.0, 2.0)
    matrix[1, 2] += rng.uniform(-2.0, 2.0)

    # Warp affine with reflection padding to avoid black borders
    img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 4. Random brightness and contrast adjustment
    alpha = rng.uniform(0.85, 1.15)  # Contrast
    beta = rng.uniform(-10.0, 10.0)  # Brightness
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return img


def balance_training_set(train_dir: Path, target_count: int | None = None, seed: int = 42) -> dict:
    rng = random.Random(seed)
    current_counts = {}
    class_files = {}

    for emotion in EMOTIONS:
        folder = train_dir / emotion
        files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        current_counts[emotion] = len(files)
        class_files[emotion] = files

    max_count = max(current_counts.values())
    target = target_count if target_count is not None else max_count
    print(f"Target count per emotion: {target} (Current maximum: {max_count})")

    augmented_counts = {}
    new_totals = {}

    for emotion in EMOTIONS:
        existing = class_files[emotion]
        count = len(existing)
        needed = target - count
        folder = train_dir / emotion

        if needed <= 0:
            print(f"  [{emotion}] {count} images (Already >= target, skipping)")
            augmented_counts[emotion] = 0
            new_totals[emotion] = count
            continue

        print(f"  [{emotion}] {count} -> {target} (+{needed} augmented images)...")
        aug_idx = 0
        while aug_idx < needed:
            # Pick a random base image to augment
            src_file = rng.choice(existing)
            img = cv2.imread(str(src_file), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            aug_img = augment_image(img, rng)
            aug_filename = folder / f"aug_{aug_idx:05d}_{src_file.stem}.png"
            cv2.imwrite(str(aug_filename), aug_img)
            aug_idx += 1

        augmented_counts[emotion] = needed
        new_totals[emotion] = target

    return {"before": current_counts, "added": augmented_counts, "after": new_totals}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "fer2013")
    ap.add_argument("--target", type=int, default=None, help="Target count per emotion (default: max class count)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_dir = args.data / "train"
    if not train_dir.is_dir():
        print(f"Error: Train directory not found at {train_dir}")
        return 1

    print("Balancing training set via facial emotion-preserving data augmentation...")
    res = balance_training_set(train_dir, args.target, args.seed)

    # Update manifest if present
    manifest_path = args.data / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["counts"]["train"] = res["after"]
            manifest["totals"]["train"] = sum(res["after"].values())
            manifest["balanced_augmentation"] = {
                "added_per_class": res["added"],
                "target_per_class": max(res["after"].values()),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print("Updated data/fer2013/manifest.json")
        except Exception as exc:
            print(f"Notice: manifest update failed ({exc})")

    print("\nAugmentation Complete:")
    for emo in EMOTIONS:
        print(f"  - {emo:<10}: {res['before'][emo]:>5} -> {res['after'][emo]:>5} (+{res['added'][emo]})")
    print(f"Total training images: {sum(res['after'].values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
