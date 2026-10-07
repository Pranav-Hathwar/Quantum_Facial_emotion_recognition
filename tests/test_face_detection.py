import cv2
import numpy as np
import pytest

from ml.preprocessing.face_detector import FaceDetector


def _sample_face():
    data = pytest.importorskip("skimage.data")
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)  # public-domain NASA photo bundled with scikit-image


def test_blank_image_has_no_faces():
    assert FaceDetector("haar").detect(np.zeros((300, 300, 3), np.uint8)) == []


def test_detects_face_and_returns_bounding_box():
    img = _sample_face()
    faces = FaceDetector("haar").detect(img)
    assert len(faces) >= 1
    f = faces[0]
    H, W = img.shape[:2]
    assert f.face_id == 1 and 0 <= f.x < W and 0 <= f.y < H and f.width > 24 and f.height > 24
    assert f.x + f.width <= W and f.y + f.height <= H
    d = f.to_dict()
    assert set(d["bounding_box"]) == {"x", "y", "width", "height"}


def test_multiple_faces_get_unique_ids():
    img = _sample_face()
    h, w = img.shape[:2]
    canvas = np.hstack([img, img, img])  # three copies side by side
    faces = FaceDetector("haar").detect(canvas)
    assert len(faces) >= 3
    ids = [f.face_id for f in faces]
    assert ids == list(range(1, len(faces) + 1))
    assert [f.x for f in faces] == sorted(f.x for f in faces)


def test_invalid_backend():
    with pytest.raises(ValueError):
        FaceDetector("nonsense")


def test_yunet_backend_if_model_available():
    from ml.preprocessing.face_detector import DEFAULT_YUNET_PATH
    if not DEFAULT_YUNET_PATH.is_file():
        pytest.skip("YuNet model not downloaded (python scripts/fetch_face_model.py)")
    faces = FaceDetector("yunet").detect(_sample_face())
    assert len(faces) >= 1 and faces[0].confidence > 0.7
