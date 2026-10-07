import zipfile

import cv2
import numpy as np

from ml.preprocessing.dataset import FER2013Dataset
from ml.preprocessing.zip_to_folders import import_zip, _route


def _jpg():
    return cv2.imencode(".jpg", np.full((48, 48), 100, np.uint8))[1].tobytes()


def test_route_splits_by_filename_prefix():
    assert _route("train/angry/Training_1.jpg") == ("train", "angry", "Training_1.jpg")
    assert _route("test/happy/PublicTest_5.jpg")[0] == "validation"
    assert _route("test/happy/PrivateTest_5.jpg")[0] == "test"
    assert _route("test/unknown/PrivateTest_5.jpg") is None
    assert _route("../evil/angry/Training_1.jpg") is None


def test_import_zip_end_to_end(tmp_path):
    zp = tmp_path / "a.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for name in ["train/sad/Training_1.jpg", "train/sad/Training_2.jpg", "test/fear/PublicTest_3.jpg",
                     "test/fear/PrivateTest_4.jpg", "test/neutral/PrivateTest_9.jpg", "readme.txt"]:
            zf.writestr(name, _jpg() if name.endswith(".jpg") else b"x")
    out = tmp_path / "fer"
    s = import_zip(zp, out)
    assert s["totals"] == {"train": 2, "validation": 1, "test": 2}
    ds = FER2013Dataset(out, "test")
    assert len(ds) == 2 and ds[0][0].shape == (48, 48)
