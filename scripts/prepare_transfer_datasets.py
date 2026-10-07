"""Standardize AffectNet, RAF-DB, and SFEW 2.0 into the canonical 7-emotion split format:
    <out>/train/<emotion>/*.(jpg|png)
    <out>/validation/<emotion>/*.(jpg|png)
    <out>/test/<emotion>/*.(jpg|png)

Canonical 7 emotions (matching ml.labels.EMOTIONS):
    0: angry, 1: disgust, 2: fear, 3: happy, 4: sad, 5: surprise, 6: neutral
"""
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1. AffectNet mapping (0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt)
AFFECTNET_MAP = {
    "0": "neutral",
    "1": "happy",
    "2": "sad",
    "3": "surprise",
    "4": "fear",
    "5": "disgust",
    "6": "angry",
    # "7" (contempt) skipped
}

# 2. RAF-DB mapping (1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral)
RAF_DB_MAP = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral",
}

# 3. SFEW mapping (Folder names to lowercase canonical)
SFEW_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def prepare_affectnet():
    print("\n[1/3] Preparing AffectNet Aligned Subset...")
    src_root = ROOT / "data" / "affectnet" / "AffectNetCustom"
    dst_root = ROOT / "data" / "affectnet_prepared"
    
    if dst_root.exists() and any(dst_root.iterdir()):
        print("      AffectNet already prepared at:", dst_root)
        return
        
    t0 = time.time()
    count = 0
    split_map = {"train": "train", "val": "validation", "test": "test"}
    
    for src_split, dst_split in split_map.items():
        split_dir = src_root / src_split
        if not split_dir.exists():
            continue
        for class_id, emotion in AFFECTNET_MAP.items():
            class_dir = split_dir / class_id
            if not class_dir.exists():
                continue
            for img in class_dir.glob("*.jpg"):
                link_or_copy(img, dst_root / dst_split / emotion / img.name)
                count += 1
                
    print(f"      [DONE] Prepared {count} AffectNet images in {time.time()-t0:.1f}s -> {dst_root}")

def prepare_raf_db():
    print("\n[2/3] Preparing RAF-DB...")
    src_root = ROOT / "data" / "raf_db" / "DATASET"
    dst_root = ROOT / "data" / "raf_db_prepared"
    
    if dst_root.exists() and any(dst_root.iterdir()):
        print("      RAF-DB already prepared at:", dst_root)
        return
        
    t0 = time.time()
    count = 0
    
    # Process train
    for class_id, emotion in RAF_DB_MAP.items():
        class_dir = src_root / "train" / class_id
        if not class_dir.exists():
            continue
        for img in class_dir.glob("*.jpg"):
            link_or_copy(img, dst_root / "train" / emotion / img.name)
            count += 1
            
    # Process test -> we populate both validation and test with RAF-DB test set
    for class_id, emotion in RAF_DB_MAP.items():
        class_dir = src_root / "test" / class_id
        if not class_dir.exists():
            continue
        for img in class_dir.glob("*.jpg"):
            link_or_copy(img, dst_root / "validation" / emotion / img.name)
            link_or_copy(img, dst_root / "test" / emotion / img.name)
            count += 2
            
    print(f"      [DONE] Prepared {count} RAF-DB images in {time.time()-t0:.1f}s -> {dst_root}")

def prepare_sfew():
    print("\n[3/3] Preparing SFEW 2.0 (Surveillance Transfer Benchmark)...")
    src_root = ROOT / "data" / "sfew"
    dst_root = ROOT / "data" / "sfew_prepared"
    
    if dst_root.exists() and any(dst_root.iterdir()):
        print("      SFEW 2.0 already prepared at:", dst_root)
        return
        
    t0 = time.time()
    count = 0
    
    # SFEW Train -> train
    train_dir = src_root / "Train"
    if train_dir.exists():
        for d in train_dir.iterdir():
            emo = SFEW_MAP.get(d.name.lower())
            if emo and d.is_dir():
                for img in d.glob("*.*"):
                    link_or_copy(img, dst_root / "train" / emo / img.name)
                    count += 1
                    
    # SFEW Val -> validation and test
    val_dir = src_root / "Val"
    if val_dir.exists():
        for d in val_dir.iterdir():
            emo = SFEW_MAP.get(d.name.lower())
            if emo and d.is_dir():
                for img in d.glob("*.*"):
                    link_or_copy(img, dst_root / "validation" / emo / img.name)
                    link_or_copy(img, dst_root / "test" / emo / img.name)
                    count += 2
                    
    print(f"      [DONE] Prepared {count} SFEW images in {time.time()-t0:.1f}s -> {dst_root}")

if __name__ == "__main__":
    print("=" * 70)
    print("DATASET PREPARATION FOR FACIAL EMOTION RECOGNITION & TRANSFER")
    print("=" * 70)
    prepare_affectnet()
    prepare_raf_db()
    prepare_sfew()
    print("\nALL DATASETS PREPARED AND NORMALIZED TO 7 EMOTION CLASSES!")
    print("=" * 70)
