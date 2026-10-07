import os
import sys
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent.parent

DATASETS = [
    {
        "name": "RAF-DB",
        "slug": "shuvoalok/raf-db-dataset",
        "dest": ROOT / "data" / "raf_db"
    },
    {
        "name": "AffectNet Aligned Subset",
        "slug": "yakhyokhuja/affectnetaligned",
        "dest": ROOT / "data" / "affectnet"
    },
    {
        "name": "SFEW 2.0 (Surveillance / In-The-Wild)",
        "slug": "vlntnstarodub/datasetsfew",
        "dest": ROOT / "data" / "sfew"
    }
]

def download_datasets():
    print("=" * 70)
    print("KAGGLE DATASET DOWNLOADER FOR QUANTUM VISION PIPELINE")
    print("=" * 70)
    
    t0_total = time.time()
    api = KaggleApi()
    print("[1/4] Authenticating with Kaggle API...")
    api.authenticate()
    print("      Authentication SUCCESSFUL for user: achintyak\n")
    
    for idx, item in enumerate(DATASETS, 1):
        name = item["name"]
        slug = item["slug"]
        dest = item["dest"]
        dest.mkdir(parents=True, exist_ok=True)
        
        print(f"[{idx+1}/4] Downloading {name} ({slug})...")
        print(f"      Destination: {dest}")
        t0 = time.time()
        
        try:
            api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
            elapsed = time.time() - t0
            
            # Count downloaded files
            all_files = list(dest.rglob("*"))
            image_files = [f for f in all_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
            print(f"      [DONE] Extracted in {elapsed:.1f}s ({len(image_files)} image files found)")
        except Exception as e:
            print(f"      [ERROR] Failed to download {name}: {e}")
            raise e
        print("-" * 70)
        
    print(f"\nALL DATASETS RETRIEVED SUCCESSFULLY in {time.time() - t0_total:.1f}s!")
    print("=" * 70)

if __name__ == "__main__":
    download_datasets()
