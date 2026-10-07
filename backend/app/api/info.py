"""Emotions, analytics, model info/metrics, quantum circuit, health."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from backend.app.api.deps import Principal, current_user, get_db, get_state
from backend.app.models import ModelRun
from backend.app.schemas.api import DISCLAIMER
from backend.app.services import store
from ml.labels import EMOTIONS
from ml.quantum.circuit import circuit_spec, draw_circuit

router = APIRouter(prefix="/api", tags=["info"])

_COLORS = {"angry": "#e66767", "disgust": "#008300", "fear": "#9085e9", "happy": "#c98500",
           "sad": "#3987e5", "surprise": "#d95926", "neutral": "#199e70"}


@router.get("/health")
def health(state=Depends(get_state)):
    try:
        with state.db.engine.connect() as c:
            c.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok, "models_available": state.engine.available_kinds(),
            "auth_enabled": not state.settings.auth_disabled, "store_images": state.settings.store_images}


@router.get("/emotions")
def emotions():
    return {"emotions": [{"index": i, "key": e, "label": e.capitalize(), "color": _COLORS[e]} for i, e in enumerate(EMOTIONS)],
            "count": len(EMOTIONS), "disclaimer": DISCLAIMER}


@router.get("/analytics")
def get_analytics(camera_id: str | None = None, minutes: int = Query(60, ge=1, le=60 * 24 * 7),
                  bucket_seconds: int = Query(60, ge=5, le=3600), _: Principal = Depends(current_user),
                  db: Session = Depends(get_db)):
    return store.analytics(db, camera_id, minutes, bucket_seconds)


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


@router.get("/model/info")
def model_info(_: Principal = Depends(current_user), state=Depends(get_state)):
    eng = state.engine
    out = {"available_models": eng.available_kinds(), "detector": None, "loaded": [],
           "pipeline": [{"step": s, "domain": d} for s, d in [
               ("Face detection (OpenCV)", "classical"), ("Preprocessing", "classical"),
               ("ResNet50 feature extraction (ImageNet transfer learning)", "classical"),
               ("Dimensionality reduction (Linear 2048 -> n_qubits)", "classical"),
               ("Quantum feature encoding (R_y angle encoding)", "quantum"),
               ("Variational quantum circuit (RX/RY/RZ + CNOT ring)", "quantum"),
               ("Quantum measurement (Pauli-Z expectation values)", "quantum"),
               ("Feature fusion (concat classical + quantum)", "classical"),
               ("Dense classifier + softmax (7 emotions)", "classical")]],
           "dataset": "FER2013 (48x48 grayscale, 7 classes)", "disclaimer": DISCLAIMER,
           "quantum_backend": "default.qubit (classical simulator)"}
    for kind in eng.available_kinds():
        try:
            b = eng.get_bundle(kind)
            out["loaded"].append({"kind": kind, "run": b.run_name, "mode": b.meta.get("mode"),
                                  "hyperparameters": b.meta.get("hyperparameters"),
                                  "head_parameters": b.meta.get("head_parameters"), "seed": b.meta.get("seed"),
                                  "smoke_test_model": b.is_smoke_test})
        except Exception as exc:  # noqa: BLE001
            out["loaded"].append({"kind": kind, "error": str(exc)})
    return out


@router.get("/model/metrics")
def model_metrics(_: Principal = Depends(current_user), state=Depends(get_state), db: Session = Depends(get_db)):
    """Measured evaluation results only (ml/models/comparison.json written by compare.py)."""
    comparison = _read_json(state.settings.models_dir / "comparison.json")
    if comparison is None or not comparison.get("table"):
        return {"available": False,
                "message": "No evaluation results yet. Train and evaluate the models, then run compare.py. "
                           "Nothing is shown until real metrics exist.", "comparison": None, "runs": []}
    runs = [{"model_name": r.model_name, "version": r.model_version, "accuracy": r.accuracy, "precision": r.precision,
             "recall": r.recall, "f1": r.f1_score, "inference_ms": r.inference_time, "created_at": r.created_at}
            for r in db.scalars(select(ModelRun).order_by(ModelRun.created_at.desc())).all()]
    return {"available": True, "comparison": comparison, "runs": runs}


@router.get("/quantum/circuit")
def quantum_circuit(n_qubits: int | None = Query(None, ge=1, le=8), n_layers: int | None = Query(None, ge=1, le=6),
                    _: Principal = Depends(current_user), state=Depends(get_state)):
    nq, nl = n_qubits, n_layers
    trained = None
    if nq is None or nl is None:
        try:
            b = state.engine.get_bundle("hybrid")
            hp = b.meta.get("hyperparameters", {})
            nq, nl = nq or hp.get("n_qubits", 4), nl or hp.get("n_layers", 2)
            trained = {"run": b.run_name, "weights": b.head.quantum.weights.detach().cpu().tolist()}
        except Exception:  # noqa: BLE001
            nq, nl = nq or 4, nl or 2
    return {**circuit_spec(nq, nl), "backend": "default.qubit (classical state-vector simulator)",
            "diagram": draw_circuit(nq, nl), "trained": trained,
            "note": "Executed on a classical simulator; no quantum hardware or quantum speed-up is involved."}
