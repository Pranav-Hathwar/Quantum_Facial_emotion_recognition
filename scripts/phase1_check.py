"""Phase 1 verification: labels, image loading, face detection, preprocessing, (optional) dataset.

    python scripts/phase1_check.py                      # uses a bundled sample face (needs scikit-image)
    python scripts/phase1_check.py --image my_photo.jpg
    python scripts/phase1_check.py --dataset data/fer2013   # also checks the converted FER2013 folders
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from ml.config import get_settings  # noqa: E402
from ml.labels import EMOTIONS  # noqa: E402
from ml.preprocessing.dataset import FER2013Dataset  # noqa: E402
from ml.preprocessing.face_detector import FaceDetector  # noqa: E402
from ml.preprocessing.image_preprocessor import crop_face, load_image, preprocess_face  # noqa: E402

failures = 0


def step(ok: bool, text: str) -> None:
    global failures
    failures += 0 if ok else 1
    print(("[PASS] " if ok else "[FAIL] ") + text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    ap.add_argument("--dataset", type=Path, default=None)
    args = ap.parse_args()
    cfg = get_settings()

    step(len(EMOTIONS) == 7 and EMOTIONS[6] == "neutral", f"7 emotion labels mapped: {', '.join(EMOTIONS)}")

    image_path = args.image
    if image_path is None:
        try:
            from skimage import data
            image_path = ROOT / "ml" / "models" / "_sample_face.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
        except ImportError:
            print("No --image given and scikit-image not installed; skipping image steps.")
            return 1
    img = load_image(image_path)
    step(img.ndim == 3, f"image loaded: {image_path.name} shape={img.shape}")

    det = FaceDetector(cfg.face_detector)
    faces = det.detect(img)
    step(len(faces) >= 1, f"face detection ({det.backend}): {len(faces)} face(s) -> {[f.bbox for f in faces]}")

    if faces:
        x = preprocess_face(crop_face(img, faces[0].bbox), cfg.image_size)
        step(x.shape == (3, cfg.image_size, cfg.image_size), f"preprocessed face tensor shape {x.shape}, dtype {x.dtype}")
        out = ROOT / "ml" / "models" / "_check_faces.jpg"
        vis = img.copy()
        for f in faces:
            cv2.rectangle(vis, (f.x, f.y), (f.x + f.width, f.y + f.height), (0, 255, 0), 2)
            cv2.putText(vis, f"Face {f.face_id}", (f.x, max(15, f.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(out), vis)
        print(f"       annotated image written to {out}")

    ds_root = args.dataset or cfg.dataset_path
    if (ds_root / "train").is_dir():
        for split in ("train", "validation", "test"):
            ds = FER2013Dataset(ds_root, split)
            step(len(ds) > 0, f"dataset {split}: {len(ds)} images, per-class {ds.class_counts()}")
    else:
        print(f"[SKIP] FER2013 not found at {ds_root} (run csv_to_folders first)")

    print("\nPhase 1 check:", "ALL PASSED" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
