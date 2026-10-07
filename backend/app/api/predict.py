from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from backend.app.api.deps import Principal, current_user, get_db, get_state, require_role
from backend.app.models import Role
from backend.app.schemas.api import PredictResponse
from backend.app.services import store
from backend.app.services.surveillance import build_response, process_frame
from backend.app.utils.images import read_image_upload
from backend.app.utils.logging import get_logger

router = APIRouter(prefix="/api/predict", tags=["prediction"])
log = get_logger("predict")


def _analyze_upload(state, db: Session, img, model: str | None) -> PredictResponse:
    analysis = state.engine.analyze(img, model)
    try:  # a database outage must never block an analysis
        store.record_predictions(db, store.new_analysis_id(), None, "upload", analysis.model.kind, analysis.faces)
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        log.error("could not store prediction", extra={"error": str(exc)[:200]})
    return build_response(analysis)


@router.post("", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), model: str | None = Query(None, description="hybrid | classical | control"),
                  _: Principal = Depends(current_user), db: Session = Depends(get_db), state=Depends(get_state)):
    """Upload an image. Top-level fields describe the largest face; `faces` lists every detected face."""
    img, _bytes = await read_image_upload(file, state.settings.max_upload_mb)
    return await run_in_threadpool(_analyze_upload, state, db, img, model)


@router.post("/multiple", response_model=PredictResponse)
async def predict_multiple(file: UploadFile = File(...), model: str | None = Query(None),
                           _: Principal = Depends(current_user), db: Session = Depends(get_db), state=Depends(get_state)):
    """Same as /predict; intended for group photos - every face gets its own prediction and id."""
    img, _bytes = await read_image_upload(file, state.settings.max_upload_mb)
    return await run_in_threadpool(_analyze_upload, state, db, img, model)


@router.post("/frame", response_model=PredictResponse)
async def predict_frame(file: UploadFile = File(...), camera_id: str = Form(...), model: str | None = Form(None),
                        _: Principal = Depends(require_role(Role.OPERATOR)), db: Session = Depends(get_db),
                        state=Depends(get_state)):
    """One webcam/camera frame: stores predictions, updates the anomaly engine, broadcasts events."""
    img, _bytes = await read_image_upload(file, state.settings.max_upload_mb)
    try:
        outcome = await run_in_threadpool(process_frame, state, db, img, camera_id, model)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    for ev in outcome.events:
        await state.ws.broadcast(camera_id, ev)
    return outcome.response
