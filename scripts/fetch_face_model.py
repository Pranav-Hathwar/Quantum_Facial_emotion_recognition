"""Optional: download the YuNet face-detection ONNX model (~230 KB) from the OpenCV Zoo.

We never download anything automatically; run this yourself if you want the better detector:
    python scripts/fetch_face_model.py
Without it, the Haar-cascade fallback is used.
"""
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DEST = Path(__file__).resolve().parents[1] / "ml" / "models" / "face_detection_yunet_2023mar.onnx"

if __name__ == "__main__":
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}\n -> {DEST}")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}")
        sys.exit(1)
    print(f"OK ({DEST.stat().st_size} bytes)")
