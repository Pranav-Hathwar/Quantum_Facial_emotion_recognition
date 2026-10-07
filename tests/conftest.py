import csv
import numpy as np
import pytest

from ml.labels import NUM_CLASSES


@pytest.fixture()
def synthetic_csv(tmp_path):
    """A tiny fake fer2013.csv: 2 rows per class per Usage split (never real data)."""
    rng = np.random.default_rng(0)
    path = tmp_path / "fer2013.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["emotion", "pixels", "Usage"])
        for usage in ("Training", "PublicTest", "PrivateTest"):
            for label in range(NUM_CLASSES):
                for _ in range(2):
                    w.writerow([label, " ".join(map(str, rng.integers(0, 256, 2304))), usage])
        w.writerow([3, "1 2 3", "Training"])      # bad pixel count -> must be skipped
        w.writerow([9, " ".join(["0"] * 2304), "Training"])  # bad label -> must be skipped
    return path


# ---------------------------------------------------------------------------------------------------------------
# backend fixtures
# ---------------------------------------------------------------------------------------------------------------
import json  # noqa: E402

import cv2  # noqa: E402


def make_run(models_dir, kind, f1=0.5, pretrained=True):
    """Write a (random-weight) checkpoint in the layout train_*.py produces so the API can serve it in tests."""
    torch = pytest.importorskip("torch")
    from ml.model_factory import build_head

    hp = {"n_qubits": 4, "n_layers": 1, "backend": "default.qubit"} if kind != "classical" else {}
    torch.manual_seed(0)
    head = build_head(kind, hp)
    run = models_dir / "runs" / f"{kind}_test"
    run.mkdir(parents=True, exist_ok=True)
    meta = {"kind": kind, "mode": "feature_extractor", "hyperparameters": hp, "seed": 0, "backbone_pretrained": pretrained}
    torch.save({"state_dict": head.state_dict(), "meta": meta, "epoch": 1}, run / "best.pt")
    (run / "history.json").write_text(json.dumps({"meta": meta, "best_val_macro_f1": f1}))
    return run


@pytest.fixture()
def face_image_bytes():
    data = pytest.importorskip("skimage.data")
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture()
def blank_image_bytes():
    return cv2.imencode(".png", np.zeros((200, 200, 3), np.uint8))[1].tobytes()


@pytest.fixture()
def make_client(tmp_path):
    from fastapi.testclient import TestClient

    from backend.app.config import AppSettings
    from backend.app.main import create_app

    clients = []

    def _make(kinds=("hybrid", "classical"), **overrides):
        models = tmp_path / "models"
        models.mkdir(exist_ok=True)
        for k in kinds:
            make_run(models, k)
        settings = AppSettings(database_url=f"sqlite:///{tmp_path / 'test.db'}", model_path=str(models),
                               pretrained_backbone=False, image_size=64, jwt_secret="test-secret",
                               admin_email="admin@test.io", admin_password="adminpass1", face_detector="haar",
                               anomaly_baseline_warmup=10, **overrides)
        app = create_app(settings)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def auth_headers(client, email="admin@test.io", password="adminpass1"):
    r = client.post("/api/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
