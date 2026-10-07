"""One frame in -> predictions stored -> anomaly engine -> events out (shared by REST and WebSocket)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from backend.app.models import Camera
from backend.app.schemas.api import AlertOut, PredictResponse
from backend.app.services import store
from backend.app.utils.logging import get_logger
from ml.exceptions import NoFaceDetectedError

log = get_logger("surveillance")


@dataclass
class FrameOutcome:
    response: PredictResponse
    events: list[dict] = field(default_factory=list)


def build_response(analysis, camera_id: str | None = None) -> PredictResponse:
    faces = analysis.faces
    primary = max(faces, key=lambda f: f.bounding_box.width * f.bounding_box.height) if faces else None
    return PredictResponse(
        model=analysis.model.run_name, model_kind=analysis.model.kind, image_width=analysis.width,
        image_height=analysis.height, face_count=len(faces), faces=faces,
        prediction=primary.emotion if primary else None, confidence=primary.confidence if primary else None,
        probabilities=primary.probabilities if primary else None, pipeline=analysis.pipeline,
        timing_ms=analysis.timing_ms, warnings=analysis.warnings, camera_id=camera_id)


def process_frame(state: Any, db: Session, img_bgr: np.ndarray, camera_id: str, model: str | None,
                  store_image_bytes: bytes | None = None) -> FrameOutcome:
    cam = db.get(Camera, camera_id)
    if cam is None:
        raise LookupError(f"Camera '{camera_id}' not found.")
    engine, alerts = state.engine, state.alerts
    events: list[dict] = []
    try:
        analysis = engine.analyze(img_bgr, model)
        response = build_response(analysis, camera_id)
    except NoFaceDetectedError:
        bundle = engine.get_bundle(model)
        from backend.app.services.inference import Analysis, _pipeline  # local import: avoid cycle at module load
        h, w = img_bgr.shape[:2]
        analysis = Analysis(bundle, w, h, [], _pipeline(bundle, {}), {}, [])
        response = build_response(analysis, camera_id)

    ts_wall = store.now()
    cam.last_active, cam.status = ts_wall, "ONLINE"
    db.commit()
    if analysis.faces:
        store.record_predictions(db, store.new_analysis_id(), camera_id, "live", analysis.model.kind, analysis.faces)
        events.append({"type": "face_detected", "camera_id": camera_id, "count": len(analysis.faces)})
        for f in analysis.faces:
            events.append({"type": "emotion_prediction", "camera_id": camera_id, "face_id": f.face_id,
                           "emotion": f.emotion, "confidence": f.confidence,
                           "bounding_box": f.bounding_box.model_dump(), "probabilities": f.probabilities})
    alert, event, result = alerts.handle_frame(db, camera_id, time.time(), [f.emotion for f in analysis.faces])
    if alert is not None and event is not None:
        out = AlertOut.model_validate(alert)
        events.append({"type": event, "camera_id": camera_id, "alert": out.model_dump(mode="json")})
        response.alert = out
    elif alert is not None:
        response.alert = AlertOut.model_validate(alert)
    events.append({"type": "camera_status", **store.camera_out(db, cam).model_dump(mode="json")})
    return FrameOutcome(response, events)
