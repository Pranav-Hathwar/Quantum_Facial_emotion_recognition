"""Import the FER2013 *image-folder* archive (e.g. Kaggle 'archive.zip') into data/fer2013/.

The archive has train/<emotion>/Training_*.jpg and test/<emotion>/{PublicTest,PrivateTest}_*.jpg.
The original test/ folder merges the dataset's PublicTest and PrivateTest sets, so we split it back using the filename prefix:
    Training    -> train        (28,709)
    PublicTest  -> validation   ( 3,589)
    PrivateTest -> test         ( 3,589)

    python -m ml.preprocessing.zip_to_folders --zip path/to/archive.zip [--out data/fer2013]
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from ml.config import PROJECT_ROOT
from ml.labels import EMOTIONS

PREFIX_TO_SPLIT = {"training": "train", "publictest": "validation", "privatetest": "test"}
SPLITS = ("train", "validation", "test")
_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _route(member: str) -> tuple[str, str, str] | None:
    """zip member path -> (split, emotion, filename), or None if it is not a FER image."""
    parts = PurePosixPath(member).parts
    if len(parts) < 3 or ".." in parts:
        return None
    emotion, filename = parts[-2].lower(), parts[-1]
    if emotion not in EMOTIONS or PurePosixPath(filename).suffix.lower() not in _IMG_EXT:
        return None
    split = PREFIX_TO_SPLIT.get(filename.split("_")[0].lower())
    if split is None:  # no recognisable prefix: fall back to the top-level folder name
        split = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}.get(parts[-3].lower())
    return (split, emotion, filename) if split else None


def import_zip(zip_path: Path, out_dir: Path, overwrite: bool = False) -> dict:
    zip_path, out_dir = Path(zip_path), Path(out_dir)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Archive not found: {zip_path}")
    for s in SPLITS:
        for e in EMOTIONS:
            (out_dir / s / e).mkdir(parents=True, exist_ok=True)
    counts = {s: Counter() for s in SPLITS}
    skipped = Counter()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            routed = _route(info.filename)
            if routed is None:
                skipped["not_a_fer_image"] += 1
                continue
            split, emotion, filename = routed
            target = out_dir / split / emotion / Path(filename).name  # basename only: no path traversal
            if target.exists() and not overwrite:
                skipped["already_exists"] += 1
            else:
                target.write_bytes(zf.read(info))
            counts[split][emotion] += 1
    summary = {
        "source": str(zip_path), "output_dir": str(out_dir),
        "counts": {s: {e: counts[s][e] for e in EMOTIONS} for s in SPLITS},
        "totals": {s: sum(counts[s].values()) for s in SPLITS},
        "skipped": dict(skipped),
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "fer2013")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    s = import_zip(a.zip, a.out, a.overwrite)
    print(json.dumps({"totals": s["totals"], "skipped": s["skipped"]}, indent=2))
    for split in SPLITS:
        print(split, s["counts"][split])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
