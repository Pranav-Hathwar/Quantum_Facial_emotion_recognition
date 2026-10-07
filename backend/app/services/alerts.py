"""Alert lifecycle on top of the anomaly detector: create / escalate / acknowledge / resolve."""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Alert, AlertStatus
from backend.app.services.anomaly import SEVERITY_ORDER, AnomalyDetector, AnomalyResult
from backend.app.services.store import aware, now
from backend.app.utils.logging import get_logger

log = get_logger("alerts")


class AlertService:
    def __init__(self, detector: AnomalyDetector, cooldown_s: float = 60.0):
        self.detector = detector
        self.cooldown_s = cooldown_s

    def handle_frame(self, db: Session, camera_id: str, ts: float, emotions: list[str]) -> tuple[Alert | None, str | None, AnomalyResult]:
        result = self.detector.observe(camera_id, ts, emotions)
        open_alert = db.scalars(select(Alert).where(Alert.camera_id == camera_id, Alert.status != AlertStatus.RESOLVED.value)
                                .order_by(Alert.timestamp.desc())).first()
        if not result.is_anomalous:
            return open_alert, None, result

        if open_alert is not None:
            escalated = SEVERITY_ORDER[result.severity] > SEVERITY_ORDER[open_alert.severity]
            open_alert.anomaly_score = max(open_alert.anomaly_score, result.score)
            open_alert.emotion_distribution = result.window_distribution
            open_alert.face_count = result.mean_faces
            open_alert.duration_s = max(open_alert.duration_s, result.duration_s)
            open_alert.updated_at = now()
            if escalated:
                open_alert.severity = result.severity
            db.commit()
            return open_alert, ("alert_updated" if escalated else None), result

        last = db.scalars(select(Alert).where(Alert.camera_id == camera_id).order_by(Alert.updated_at.desc())).first()
        if last is not None and (now() - aware(last.updated_at)) < timedelta(seconds=self.cooldown_s):
            return None, None, result

        alert = Alert(camera_id=camera_id, alert_type="EMOTIONAL_ANOMALY", severity=result.severity,
                      anomaly_score=result.score, emotion_distribution=result.window_distribution,
                      baseline_distribution=result.baseline_distribution, face_count=result.mean_faces,
                      duration_s=result.duration_s, status=AlertStatus.NEW.value)
        db.add(alert)
        db.commit()
        log.warning("emotional anomaly alert created", extra={"camera_id": camera_id, "severity": alert.severity,
                                                              "score": round(alert.anomaly_score, 3), "alert_id": alert.id})
        return alert, "alert_created", result

    @staticmethod
    def transition(db: Session, alert: Alert, new_status: AlertStatus, user_email: str, note: str | None) -> Alert:
        allowed = {AlertStatus.NEW: {AlertStatus.ACKNOWLEDGED, AlertStatus.RESOLVED},
                   AlertStatus.ACKNOWLEDGED: {AlertStatus.RESOLVED}}
        cur = AlertStatus(alert.status)
        if new_status not in allowed.get(cur, set()):
            raise ValueError(f"Cannot change alert from {cur.value} to {new_status.value}.")
        alert.status = new_status.value
        alert.handled_by = user_email
        alert.operator_action = (f"{alert.operator_action}\n" if alert.operator_action else "") + \
            f"[{new_status.value}] {note or ''}".strip()
        alert.updated_at = now()
        db.commit()
        return alert
