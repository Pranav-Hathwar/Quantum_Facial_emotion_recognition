from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import Principal, current_user, get_db, get_state, require_role
from backend.app.models import Alert, AlertStatus, Role
from backend.app.schemas.api import AlertAction, AlertOut
from backend.app.services.alerts import AlertService

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertOut])
def list_alerts(status: str | None = Query(None), camera_id: str | None = None, limit: int = Query(100, le=500),
                _: Principal = Depends(current_user), db: Session = Depends(get_db)):
    q = select(Alert)
    if status:
        if status not in {s.value for s in AlertStatus}:
            raise HTTPException(422, f"status must be one of {[s.value for s in AlertStatus]}")
        q = q.where(Alert.status == status)
    if camera_id:
        q = q.where(Alert.camera_id == camera_id)
    return db.scalars(q.order_by(Alert.timestamp.desc()).limit(limit)).all()


def _get(db: Session, alert_id: int) -> Alert:
    a = db.get(Alert, alert_id)
    if a is None:
        raise HTTPException(404, "Alert not found.")
    return a


@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(alert_id: int, _: Principal = Depends(current_user), db: Session = Depends(get_db)):
    return _get(db, alert_id)


async def _transition(alert_id: int, new: AlertStatus, body: AlertAction, user: Principal, db: Session, state):
    alert = _get(db, alert_id)
    try:
        AlertService.transition(db, alert, new, user.email, body.note)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    out = AlertOut.model_validate(alert)
    await state.ws.broadcast(alert.camera_id, {"type": "alert_updated", "camera_id": alert.camera_id,
                                               "alert": out.model_dump(mode="json")})
    return out


@router.post("/{alert_id}/acknowledge", response_model=AlertOut)
async def acknowledge(alert_id: int, body: AlertAction | None = None, user: Principal = Depends(require_role(Role.OPERATOR)),
                      db: Session = Depends(get_db), state=Depends(get_state)):
    return await _transition(alert_id, AlertStatus.ACKNOWLEDGED, body or AlertAction(), user, db, state)


@router.post("/{alert_id}/resolve", response_model=AlertOut)
async def resolve(alert_id: int, body: AlertAction | None = None, user: Principal = Depends(require_role(Role.OPERATOR)),
                  db: Session = Depends(get_db), state=Depends(get_state)):
    return await _transition(alert_id, AlertStatus.RESOLVED, body or AlertAction(), user, db, state)
