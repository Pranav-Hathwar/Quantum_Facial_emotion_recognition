"""Image loading and face preprocessing (Chapter 12, step b).

Pipeline per face: crop (+margin) -> BGR->RGB -> resize to IMAGE_SIZE -> scale to [0,1]
-> ImageNet mean/std normalisation (what pretrained ResNet50 expects) -> CHW float32.
The exact same function is used for training-time feature caching and for inference.
"""
from __future__ import annotations

import numpy as np
import cv2

from ml.exceptions import InvalidImageError

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_PIXELS = 50_000_000  # guard against decompression bombs


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPG/PNG/WEBP bytes to a BGR uint8 image. Raises InvalidImageError on anything else."""
    if not data:
        raise InvalidImageError("Empty image data.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("Invalid image. Supported formats: JPG, JPEG, PNG, WEBP.")
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise InvalidImageError("Image is too large.")
    return img


def load_image(path) -> np.ndarray:
    from pathlib import Path

    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InvalidImageError(f"Unsupported file type '{p.suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    if not p.is_file():
        raise InvalidImageError(f"File not found: {p}")
    return decode_image(p.read_bytes())


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise InvalidImageError(f"Unsupported image shape {img.shape}")


def crop_face(img_bgr: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.15) -> np.ndarray:
    """Crop (x, y, w, h) with a relative margin, clipped to the image."""
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    if x1 <= x0 or y1 <= y0:
        raise InvalidImageError("Face bounding box is outside the image.")
    return img_bgr[y0:y1, x0:x1]


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Optional lighting correction (chapter step b). Off by default so train/inference stay identical."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_face(face: np.ndarray, size: int = 224, use_clahe: bool = False) -> np.ndarray:
    """One face image (gray/BGR/BGRA, any size) -> float32 array of shape (3, size, size), ImageNet-normalised."""
    if face is None or face.size == 0:
        raise InvalidImageError("Empty face crop.")
    bgr = _to_bgr(face)
    if use_clahe:
        bgr = apply_clahe(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_CUBIC if min(rgb.shape[:2]) < size else cv2.INTER_AREA  # FER2013 is 48px -> upsample
    rgb = cv2.resize(rgb, (size, size), interpolation=interp)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess_batch(faces: list[np.ndarray], size: int = 224, use_clahe: bool = False) -> np.ndarray:
    """List of faces -> (N, 3, size, size) float32."""
    return np.stack([preprocess_face(f, size, use_clahe) for f in faces], axis=0)


def to_torch_batch(batch: np.ndarray, device: str = "cpu"):
    """numpy (N,3,S,S) or (3,S,S) -> torch tensor with batch dimension. Imports torch lazily (Phase 2+)."""
    import torch

    t = torch.from_numpy(batch)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t.to(device)
