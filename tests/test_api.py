import pytest

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from tests.conftest import auth_headers  # noqa: E402

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def _post_image(client, path, data, headers, **kw):
    return client.post(path, files={"file": ("face.jpg", data, "image/jpeg")}, headers=headers, **kw)


def test_health_and_emotions_endpoints(make_client):
    c = make_client()
    assert c.get("/api/health").json()["database"] is True
    h = auth_headers(c)
    body = c.get("/api/emotions").json()
    assert body["count"] == 7 and [e["key"] for e in body["emotions"]] == EMOTIONS
    assert "do not establish" in body["disclaimer"]


def test_requires_authentication(make_client):
    c = make_client()
    assert c.get("/api/cameras").status_code == 401
    assert c.post("/api/auth/login", data={"username": "admin@test.io", "password": "wrong"}).status_code == 401


def test_predict_returns_seven_probabilities_and_hybrid_pipeline(make_client, face_image_bytes):
    c = make_client()
    h = auth_headers(c)
    r = _post_image(c, "/api/predict?model=hybrid", face_image_bytes, h)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["model_kind"] == "hybrid" and d["face_count"] >= 1
    assert list(d["probabilities"]) == EMOTIONS and abs(sum(d["probabilities"].values()) - 1) < 1e-2
    assert d["prediction"] in EMOTIONS and 0 <= d["confidence"] <= 1
    face = d["faces"][0]
    assert set(face["bounding_box"]) == {"x", "y", "width", "height"} and len(face["quantum_features"]) == 4
    quantum = {s["stage"]: s["status"] for s in d["pipeline"]}
    assert quantum["vqc"] == "ok" and quantum["quantum_encoding"] == "ok"
    assert "probabilistic" in d["disclaimer"] and any("smoke test" in w for w in d["warnings"]) is False


def test_classical_model_has_no_quantum_stages(make_client, face_image_bytes):
    c = make_client()
    d = _post_image(c, "/api/predict?model=classical", face_image_bytes, auth_headers(c)).json()
    assert d["faces"][0]["quantum_features"] is None
    assert {s["stage"]: s["status"] for s in d["pipeline"]}["vqc"] == "not_applicable"


def test_multiple_faces_each_get_a_prediction(make_client, face_image_bytes):
    import cv2, numpy as np
    img = cv2.imdecode(np.frombuffer(face_image_bytes, np.uint8), 1)
    three = cv2.imencode(".jpg", np.hstack([img, img, img]))[1].tobytes()
    c = make_client()
    d = _post_image(c, "/api/predict/multiple", three, auth_headers(c)).json()
    assert d["face_count"] >= 3
    assert [f["face_id"] for f in d["faces"]] == list(range(1, d["face_count"] + 1))
    assert all(len(f["probabilities"]) == 7 for f in d["faces"])


def test_error_cases_are_clear_not_crashes(make_client, blank_image_bytes):
    c = make_client()
    h = auth_headers(c)
    r = _post_image(c, "/api/predict", blank_image_bytes, h)
    assert r.status_code == 422 and "No face detected" in r.json()["detail"]
    r = c.post("/api/predict", files={"file": ("x.jpg", b"garbage", "image/jpeg")}, headers=h)
    assert r.status_code == 400 and "Invalid image" in r.json()["detail"]
    r = c.post("/api/predict", files={"file": ("x.pdf", b"%PDF", "application/pdf")}, headers=h)
    assert r.status_code == 400


def test_model_unavailable_gives_503(make_client, face_image_bytes):
    c = make_client(kinds=())
    r = _post_image(c, "/api/predict", face_image_bytes, auth_headers(c))
    assert r.status_code == 503 and r.json()["code"] == "model_unavailable"
    assert c.get("/api/model/metrics", headers=auth_headers(c)).json()["available"] is False


def test_roles_are_enforced(make_client, face_image_bytes):
    c = make_client()
    admin = auth_headers(c)
    for role in ("OPERATOR", "VIEWER"):
        r = c.post("/api/auth/users", json={"name": role, "email": f"{role.lower()}@test.io", "password": "password123", "role": role}, headers=admin)
        assert r.status_code == 201
    op, viewer = auth_headers(c, "operator@test.io", "password123"), auth_headers(c, "viewer@test.io", "password123")
    assert c.post("/api/cameras", json={"name": "New"}, headers=viewer).status_code == 403
    assert c.post("/api/cameras", json={"name": "New"}, headers=op).status_code == 403
    assert c.post("/api/cameras", json={"name": "New", "location": "Gate"}, headers=admin).status_code == 201
    assert c.get("/api/cameras", headers=viewer).status_code == 200
    assert c.post("/api/auth/users", json={"name": "x", "email": "x@test.io", "password": "password123"}, headers=op).status_code == 403
    r = c.post("/api/predict/frame", files={"file": ("f.jpg", face_image_bytes, "image/jpeg")}, data={"camera_id": "CAM-001"}, headers=viewer)
    assert r.status_code == 403
    assert c.post("/api/predict/frame", files={"file": ("f.jpg", face_image_bytes, "image/jpeg")}, data={"camera_id": "NOPE"}, headers=op).status_code == 404


def test_frame_pipeline_stores_predictions_and_feeds_analytics(make_client, face_image_bytes):
    c = make_client()
    h = auth_headers(c)
    for _ in range(2):
        r = c.post("/api/predict/frame", files={"file": ("f.jpg", face_image_bytes, "image/jpeg")}, data={"camera_id": "CAM-001"}, headers=h)
        assert r.status_code == 200 and r.json()["camera_id"] == "CAM-001"
    a = c.get("/api/analytics?camera_id=CAM-001", headers=h).json()
    assert a["total_analyses"] == 2 and a["total_faces_detected"] >= 2 and a["dominant_emotion"] in EMOTIONS
    assert abs(sum(v["percent"] for v in a["distribution"].values()) - 100) < 0.5
    cams = {x["id"]: x for x in c.get("/api/cameras", headers=h).json()}
    assert cams["CAM-001"]["status"] == "ONLINE" and cams["CAM-001"]["current_emotion"] in EMOTIONS


def test_quantum_circuit_and_model_endpoints(make_client):
    c = make_client()
    h = auth_headers(c)
    q = c.get("/api/quantum/circuit", headers=h).json()
    assert q["n_qubits"] == 4 and q["n_layers"] == 1 and "RY" in q["diagram"] and q["trained"] is not None
    info = c.get("/api/model/info", headers=h).json()
    assert set(info["available_models"]) == {"hybrid", "classical"} and "simulator" in info["quantum_backend"]


def test_database_down_returns_503_but_service_stays_up(make_client, tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.config import AppSettings
    from backend.app.main import create_app
    app = create_app(AppSettings(database_url="sqlite:////nonexistent_dir/x/y.db", model_path=str(tmp_path)))
    with TestClient(app) as c:
        assert c.get("/api/emotions").status_code == 200
        r = c.post("/api/auth/login", data={"username": "a@b.io", "password": "x"})
        assert r.status_code == 503 and "Database unavailable" in r.json()["detail"]
        assert c.get("/api/health").json()["status"] == "degraded"


def test_websocket_frame_roundtrip_and_viewer_cannot_send(make_client, face_image_bytes):
    c = make_client()
    admin = auth_headers(c)["Authorization"].split()[1]
    with c.websocket_connect(f"/ws/surveillance/CAM-001?token={admin}") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "connected" and hello["can_send_frames"] is True
        ws.send_bytes(face_image_bytes)
        types = []
        for _ in range(12):
            msg = ws.receive_json()
            types.append(msg["type"])
            if msg["type"] == "camera_status":
                break
        assert "frame_result" in types and "emotion_prediction" in types and types[-1] == "camera_status"
    with pytest.raises(Exception):
        with c.websocket_connect("/ws/surveillance/CAM-001?token=bad"):
            pass
