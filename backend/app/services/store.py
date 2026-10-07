"""Persistence + read-side helpers (predictions, camera state, analytics, model runs)."""
from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models import Alert, AlertStatus, Camera, EmotionPrediction, ModelRun
from backend.app.schemas.api import CameraOut, FaceResult
from backend.app.utils.logging import get_logger
from ml.labels import EMOTIONS

log = get_logger("store")
ONLINE_WINDOW_S = 15


def aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_analysis_id() -> str:
    return uuid.uuid4().hex


def record_predictions(db: Session, analysis_id: str, camera_id: str | None, source: str, model: str,
                       faces: list[FaceResult]) -> None:
    ts = now()
    db.add_all([EmotionPrediction(analysis_id=analysis_id, camera_id=camera_id, face_id=f.face_id, model_name=model,
                                  emotion=f.emotion, confidence=f.confidence, probabilities=f.probabilities,
                                  source=source, timestamp=ts) for f in faces])
    db.commit()


def camera_out(db: Session, cam: Camera) -> CameraOut:
    last = aware(cam.last_active)
    online = last is not None and (now() - last).total_seconds() <= ONLINE_WINDOW_S
    rows = db.execute(select(EmotionPrediction.emotion, EmotionPrediction.analysis_id, EmotionPrediction.timestamp)
                      .where(EmotionPrediction.camera_id == cam.id,
                             EmotionPrediction.timestamp >= now() - timedelta(seconds=30))
                      .order_by(EmotionPrediction.timestamp.desc()).limit(200)).all()
    cur, faces = None, 0
    if rows:
        latest = rows[0].analysis_id
        latest_rows = [r for r in rows if r.analysis_id == latest]
        faces = len(latest_rows)
        cur = Counter(r.emotion for r in rows).most_common(1)[0][0]
    open_alert = db.scalars(select(Alert).where(Alert.camera_id == cam.id, Alert.status != AlertStatus.RESOLVED.value)
                            .order_by(Alert.timestamp.desc())).first()
    return CameraOut(id=cam.id, name=cam.name, location=cam.location, source=cam.source,
                     status="ONLINE" if online else "OFFLINE", last_active=last,
                     current_emotion=cur if online else None, face_count=faces if online else 0,
                     alert_state=f"{open_alert.severity} ALERT" if open_alert else "No Alert")


def next_camera_id(db: Session) -> str:
    n = db.scalar(select(func.count()).select_from(Camera)) or 0
    while True:
        n += 1
        cid = f"CAM-{n:03d}"
        if db.get(Camera, cid) is None:
            return cid


def analytics(db: Session, camera_id: str | None, minutes: int, bucket_s: int) -> dict:
    since = now() - timedelta(minutes=minutes)
    q = select(EmotionPrediction).where(EmotionPrediction.timestamp >= since)
    if camera_id:
        q = q.where(EmotionPrediction.camera_id == camera_id)
    rows = db.scalars(q.order_by(EmotionPrediction.timestamp).limit(50_000)).all()

    counts = Counter(r.emotion for r in rows)
    total = len(rows)
    dist = {e: {"count": counts.get(e, 0), "percent": round(100 * counts.get(e, 0) / total, 2) if total else 0.0}
            for e in EMOTIONS}
    buckets: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        buckets[int(aware(r.timestamp).timestamp() // bucket_s) * bucket_s][r.emotion] += 1
    timeline = [{"t": datetime.fromtimestamp(t, timezone.utc).isoformat(), "total": sum(c.values()),
                 "counts": {e: c.get(e, 0) for e in EMOTIONS}} for t, c in sorted(buckets.items())]
    recent = [{"timestamp": aware(r.timestamp).isoformat(), "camera_id": r.camera_id, "face_id": r.face_id,
               "emotion": r.emotion, "confidence": r.confidence, "source": r.source} for r in rows[-50:]][::-1]

    analyses = len({r.analysis_id for r in rows})
    active_alerts = db.scalar(select(func.count()).select_from(Alert).where(Alert.status != AlertStatus.RESOLVED.value)) or 0
    cams = db.scalars(select(Camera)).all()
    connected = sum(1 for c in cams if camera_out(db, c).status == "ONLINE")
    return {
        "window_minutes": minutes, "camera_id": camera_id,
        "total_faces_detected": total, "total_analyses": analyses,
        "dominant_emotion": counts.most_common(1)[0][0] if counts else None,
        "distribution": dist, "timeline": timeline, "recent": recent,
        "active_alerts": active_alerts, "connected_cameras": connected, "total_cameras": len(cams),
    }


# ---- model runs (read from evaluation artefacts, never hand-entered) -------------------------------------------
def sync_model_runs(db: Session, models_dir: Path) -> int:
    added = 0
    for m in sorted((models_dir / "runs").glob("*/metrics.json")):
        try:
            data = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        if data.get("split") != "test" or data.get("backbone", {}).get("pretrained") is False:
            continue
        version = m.parent.name
        if db.scalar(select(ModelRun).where(ModelRun.model_version == version)):
            continue
        db.add(ModelRun(model_name=data["model_kind"], model_version=version, accuracy=data["accuracy"],
                        precision=data["macro"]["precision"], recall=data["macro"]["recall"],
                        f1_score=data["macro"]["f1"], inference_time=data["inference"].get("total_ms_per_face"),
                        details={"seed": data.get("seed"), "training": data.get("training"),
                                 "parameters": data.get("parameters")}))
        added += 1
    if added:
        db.commit()
    return added
