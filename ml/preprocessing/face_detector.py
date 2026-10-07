"""Lightweight multi-face detection (Chapter 12, step b).

Backends:
  * yunet - OpenCV's YuNet DNN detector (better; needs a ~230 KB ONNX file, see scripts/fetch_face_model.py)
  * haar  - OpenCV Haar cascade (ships with OpenCV; weaker on small/side faces)
  * auto  - yunet if the model file is present, otherwise haar
Each detected face gets a face_id (1..N, left-to-right) and a bounding box.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from ml.config import PROJECT_ROOT

DEFAULT_YUNET_PATH = PROJECT_ROOT / "ml" / "models" / "face_detection_yunet_2023mar.onnx"
_MAX_SIDE = 1280  # downscale very large frames before detection for speed


@dataclass
class DetectedFace:
    face_id: int
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def to_dict(self) -> dict:
        d = asdict(self)
        return {"face_id": d["face_id"],
                "bounding_box": {"x": d["x"], "y": d["y"], "width": d["width"], "height": d["height"]},
                "detection_confidence": round(d["confidence"], 4)}


class FaceDetector:
    def __init__(self, backend: str = "auto", yunet_model: Path | None = None,
                 score_threshold: float = 0.7, min_face_size: int = 24):
        if backend not in {"auto", "yunet", "haar"}:
            raise ValueError("backend must be 'auto', 'yunet' or 'haar'")
        self.score_threshold, self.min_face_size = score_threshold, min_face_size
        model = Path(yunet_model) if yunet_model else DEFAULT_YUNET_PATH
        has_yunet = model.is_file() and hasattr(cv2, "FaceDetectorYN")
        if backend == "yunet" and not has_yunet:
            raise FileNotFoundError(f"YuNet model not found at {model}. Run: python scripts/fetch_face_model.py")
        self.backend = "yunet" if (backend == "yunet" or (backend == "auto" and has_yunet)) else "haar"
        self._yunet = None
        self._haar = None
        if self.backend == "yunet":
            self._yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score_threshold, 0.3, 5000)
        else:
            cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(str(cascade))
            if self._haar.empty():
                raise RuntimeError(f"Could not load Haar cascade at {cascade}")

    def detect(self, img_bgr: np.ndarray) -> list[DetectedFace]:
        """Return all faces found (possibly empty). Never assumes a single person."""
        if img_bgr is None or img_bgr.size == 0:
            return []
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        H, W = img_bgr.shape[:2]
        scale = min(1.0, _MAX_SIDE / max(H, W))
        work = cv2.resize(img_bgr, (int(W * scale), int(H * scale))) if scale < 1.0 else img_bgr
        raw = self._detect_yunet(work) if self.backend == "yunet" else self._detect_haar(work)

        boxes: list[tuple[int, int, int, int, float]] = []
        for x, y, w, h, s in raw:
            x, y, w, h = (int(round(v / scale)) for v in (x, y, w, h))
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if (x1 - x0) >= self.min_face_size and (y1 - y0) >= self.min_face_size:
                boxes.append((x0, y0, x1 - x0, y1 - y0, float(s)))
        boxes.sort(key=lambda b: b[0])  # left-to-right -> stable face ids
        return [DetectedFace(i + 1, *b) for i, b in enumerate(boxes)]

    def _detect_yunet(self, img: np.ndarray):
        h, w = img.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(img)
        if faces is None:
            return []
        return [(f[0], f[1], f[2], f[3], f[14]) for f in faces]

    def _detect_haar(self, img: np.ndarray):
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        min_sz = (self.min_face_size, self.min_face_size)
        try:
            rects, _, weights = self._haar.detectMultiScale3(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=min_sz, outputRejectLevels=True)
        except cv2.error:
            rects = self._haar.detectMultiScale(gray, 1.1, 5, minSize=min_sz)
            weights = [1.0] * len(rects)
        return [(x, y, w, h, float(s)) for (x, y, w, h), s in zip(rects, weights)]
