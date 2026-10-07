import json
import pytest

from ml.labels import EMOTIONS
from ml.exceptions import DatasetNotFoundError
from ml.preprocessing.csv_to_folders import convert
from ml.preprocessing.dataset import FER2013Dataset


def test_convert_creates_layout_and_skips_bad_rows(synthetic_csv, tmp_path):
    out = tmp_path / "fer"
    summary = convert(synthetic_csv, out)
    assert summary["totals"] == {"train": 14, "validation": 14, "test": 14}
    assert summary["skipped"] == {"bad_pixels": 1, "bad_label": 1}
    for split in ("train", "validation", "test"):
        for emo in EMOTIONS:
            assert (out / split / emo).is_dir()
    assert json.loads((out / "manifest.json").read_text())["totals"]["train"] == 14


def test_dataset_loads_images_with_correct_labels(synthetic_csv, tmp_path):
    out = tmp_path / "fer"
    convert(synthetic_csv, out)
    ds = FER2013Dataset(out, "train")
    assert len(ds) == 14
    img, label = ds[0]
    assert img.shape == (48, 48) and img.dtype.name == "uint8"
    assert {l for _, l in ds.samples} == set(range(7))
    assert ds.class_counts() == {e: 2 for e in EMOTIONS}
    assert len(ds.class_weights()) == 7
    rgb, _ = FER2013Dataset(out, "test", as_rgb=True)[0]
    assert rgb.shape == (48, 48, 3)


def test_conversion_is_idempotent(synthetic_csv, tmp_path):
    out = tmp_path / "fer"
    convert(synthetic_csv, out)
    again = convert(synthetic_csv, out)
    assert again["skipped"].get("already_exists") == 42


def test_missing_dataset_gives_helpful_error(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="csv_to_folders"):
        FER2013Dataset(tmp_path / "nope", "train")
