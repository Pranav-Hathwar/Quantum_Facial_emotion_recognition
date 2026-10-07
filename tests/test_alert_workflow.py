import random

import pytest

pytest.importorskip("fastapi")
from tests.conftest import auth_headers  # noqa: E402


def _insert_alert(c, camera="CAM-002", severity="HIGH"):
    from backend.app.models import Alert
    with c.app.state.db.session() as s:
        a = Alert(camera_id=camera, severity=severity, anomaly_score=0.7, emotion_distribution={"fear": 0.6},
                  baseline_distribution={"neutral": 0.7}, face_count=6, duration_s=15)
        s.add(a); s.commit()
        return a.id


def test_alert_lifecycle_requires_operator_and_valid_transitions(make_client):
    c = make_client()
    admin = auth_headers(c)
    c.post("/api/auth/users", json={"name": "v", "email": "v@test.io", "password": "password123", "role": "VIEWER"}, headers=admin)
    c.post("/api/auth/users", json={"name": "o", "email": "o@test.io", "password": "password123", "role": "OPERATOR"}, headers=admin)
    viewer, op = auth_headers(c, "v@test.io", "password123"), auth_headers(c, "o@test.io", "password123")
    aid = _insert_alert(c)

    assert c.post(f"/api/alerts/{aid}/acknowledge", headers=viewer).status_code == 403
    r = c.post(f"/api/alerts/{aid}/acknowledge", json={"note": "checking camera"}, headers=op)
    assert r.status_code == 200 and r.json()["status"] == "ACKNOWLEDGED" and r.json()["handled_by"] == "o@test.io"
    r = c.post(f"/api/alerts/{aid}/resolve", json={"note": "crowd dispersed, false alarm"}, headers=op)
    assert r.json()["status"] == "RESOLVED" and "false alarm" in r.json()["operator_action"]
    assert c.post(f"/api/alerts/{aid}/acknowledge", headers=op).status_code == 409   # cannot go back
    assert c.get("/api/alerts?status=RESOLVED", headers=viewer).json()[0]["id"] == aid
    assert c.get("/api/alerts?status=BOGUS", headers=viewer).status_code == 422
    assert "human review" in c.get(f"/api/alerts/{aid}", headers=viewer).json()["disclaimer"]


def test_camera_alert_state_reflects_open_alert(make_client):
    c = make_client()
    h = auth_headers(c)
    aid = _insert_alert(c, "CAM-002", "HIGH")
    cams = {x["id"]: x for x in c.get("/api/cameras", headers=h).json()}
    assert cams["CAM-002"]["alert_state"] == "HIGH ALERT" and cams["CAM-001"]["alert_state"] == "No Alert"
    c.post(f"/api/alerts/{aid}/resolve", headers=h)
    assert {x["id"]: x for x in c.get("/api/cameras", headers=h).json()}["CAM-002"]["alert_state"] == "No Alert"


def test_alert_service_creates_one_alert_and_cooldown(make_client):
    """Drive the real AlertService with a calm baseline then sustained group fear, against the DB."""
    from backend.app.models import Alert
    from backend.app.services.anomaly import DISTRESS_EMOTIONS
    c = make_client()
    svc = c.app.state.alerts
    rng = random.Random(1)
    created = 0
    with c.app.state.db.session() as db:
        t = 0.0
        for _ in range(180):
            _, ev, _r = svc.handle_frame(db, "CAM-002", t, [rng.choice(["neutral", "happy"]) for _ in range(5)])
            t += 0.5
        for _ in range(80):
            _, ev, _r = svc.handle_frame(db, "CAM-002", t, [rng.choice(DISTRESS_EMOTIONS) if rng.random() < 0.8 else "neutral" for _ in range(6)])
            created += ev == "alert_created"
            t += 0.5
        alerts = db.query(Alert).filter_by(camera_id="CAM-002").all()
    assert created == 1 and len(alerts) == 1 and alerts[0].status == "NEW"
    assert alerts[0].severity in {"MEDIUM", "HIGH", "CRITICAL"} and alerts[0].emotion_distribution["fear"] > 0.1
