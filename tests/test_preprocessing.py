import cv2
import numpy as np
import pytest

from ml.exceptions import InvalidImageError
from ml.preprocessing.image_preprocessor import (
    crop_face, decode_image, preprocess_batch, preprocess_face, to_torch_batch, IMAGENET_MEAN, IMAGENET_STD)


def test_preprocess_gray_48px_to_resnet_input():
    gray = np.full((48, 48), 128, dtype=np.uint8)
    out = preprocess_face(gray, 224)
    assert out.shape == (3, 224, 224) and out.dtype == np.float32
    expected = (128 / 255 - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(out[:, 100, 100], expected, atol=1e-4)  # normalisation matches ImageNet stats


def test_color_order_bgr_to_rgb():
    bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255  # pure red in BGR
    out = preprocess_face(bgr, 32)
    red = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert np.allclose(out[0], red, atol=1e-4)          # channel 0 of RGB == red
    assert np.allclose(out[2], (0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2], atol=1e-4)


def test_batch_and_bgra_input():
    faces = [np.zeros((50, 40, 4), np.uint8), np.zeros((80, 80), np.uint8)]
    assert preprocess_batch(faces, 64).shape == (2, 3, 64, 64)


def test_crop_face_clips_to_image():
    img = np.zeros((100, 100, 3), np.uint8)
    assert crop_face(img, (90, 90, 30, 30), margin=0.5).shape[0] <= 100
    with pytest.raises(InvalidImageError):
        crop_face(img, (200, 200, 10, 10))


def test_decode_rejects_garbage_and_accepts_png_jpg():
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image")
    with pytest.raises(InvalidImageError):
        decode_image(b"")
    img = np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    for ext in (".png", ".jpg"):
        ok, buf = cv2.imencode(ext, img)
        assert decode_image(buf.tobytes()).shape == (32, 32, 3)


def test_torch_batch_adds_batch_dim():
    torch = pytest.importorskip("torch")
    t = to_torch_batch(preprocess_face(np.zeros((48, 48), np.uint8), 64))
    assert tuple(t.shape) == (1, 3, 64, 64) and t.dtype == torch.float32
