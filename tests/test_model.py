import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from ml.classical.head import ClassicalHead
from ml.evaluation.metrics import compute_metrics
from ml.model_factory import build_head, load_head
from ml.training.data import feature_loader
from ml.training.trainer import TrainConfig, fit, predict_probs


def _toy(n=210, d=32, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 7
    centers = rng.normal(size=(7, d)) * 2
    X = centers[y] + rng.normal(size=(n, d)) * 0.5
    return X.astype(np.float32), y


def test_classical_head_outputs_seven_logits():
    h = ClassicalHead(in_features=32, hidden=8)
    assert h(torch.randn(4, 32)).shape == (4, 7)


@pytest.mark.parametrize("kind", ["classical", "control", "hybrid"])
def test_each_head_learns_separable_toy_data_and_checkpoint_roundtrips(kind, tmp_path):
    X, y = _toy()
    hp = {"in_features": 32, "hidden": 16, "n_qubits": 4, "n_layers": 1}
    model = build_head(kind, hp)
    tl, vl = feature_loader(X, y, 64, True), feature_loader(X, y, 64, False)
    hist = fit(model, tl, vl, TrainConfig(epochs=12, lr=1e-2, batch_size=64, patience=50),
               tmp_path, None, {"kind": kind, "hyperparameters": hp}, log=lambda *_: None)
    assert hist["best_val_macro_f1"] > 0.6 and hist["train_time_s"] > 0
    loaded, meta = load_head(tmp_path / "best.pt")
    p1, _ = predict_probs(loaded, vl, "cpu")
    assert p1.shape == (len(y), 7) and np.allclose(p1.sum(1), 1, atol=1e-5)
    assert meta["kind"] == kind
    assert json.loads((tmp_path / "history.json").read_text())["best_epoch"] >= 1


def test_metrics_known_values():
    y = np.array([0, 0, 1, 1, 6, 6])
    p = np.array([0, 1, 1, 1, 6, 0])
    m = compute_metrics(y, p)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["angry"]["recall"] == pytest.approx(0.5)
    assert m["per_class"]["disgust"]["precision"] == pytest.approx(2 / 3)
    assert np.array(m["confusion_matrix"]).sum() == 6 and len(m["class_order"]) == 7
