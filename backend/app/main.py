"""QuantumVision API.   Run from the repo root:  uvicorn backend.app.main:app --reload"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from backend.app.api import alerts, auth, cameras, info, predict, ws
from backend.app.config import AppSettings, get_app_settings
from backend.app.database.session import Database
from backend.app.models import Camera, Role, User
from backend.app.services.alerts import AlertService
from backend.app.services.anomaly import AnomalyConfig, AnomalyDetector
from backend.app.services.inference import InferenceEngine, ModelUnavailableError, QuantumCircuitError
from backend.app.services.store import sync_model_runs
from backend.app.utils.logging import configure_logging, get_logger
from backend.app.utils.security import hash_password
from ml.exceptions import InvalidImageError, NoFaceDetectedError

log = get_logger("app")

DEMO_CAMERAS = [("CAM-001", "Laptop Webcam", "Demo desk", "webcam"),
                ("CAM-002", "Metro Station Gate 1", "Simulated feed", "video-file"),
                ("CAM-003", "Mall Entrance", "Simulated feed", "video-file")]


def seed(db: Database, s: AppSettings) -> None:
    with db.session() as session:
        if session.scalar(select(User).where(User.email == s.admin_email.lower())) is None:
            session.add(User(name="Administrator", email=s.admin_email.lower(),
                             password_hash=hash_password(s.admin_password), role=Role.ADMIN.value))
        if session.scalar(select(Camera).limit(1)) is None:
            session.add_all([Camera(id=i, name=n, location=loc, source=src) for i, n, loc, src in DEMO_CAMERAS])
        session.commit()
        try:
            sync_model_runs(session, s.models_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("model_runs sync skipped", extra={"error": str(exc)[:200]})


def create_app(settings: AppSettings | None = None) -> FastAPI:
    s = settings or get_app_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("QuantumVision starting", extra={"env": s.environment, "store_images": s.store_images,
                                                  "db": s.database_url.split("@")[-1].split("///")[-1]})
        try:
            app.state.db.create_all()
            seed(app.state.db, s)
        except SQLAlchemyError as exc:
            log.error("database unavailable at startup", extra={"error": str(exc)[:200]})
        yield
        log.info("QuantumVision stopped")

    app = FastAPI(title="QuantumVision API", version="1.0.0", lifespan=lifespan,
                  description="Hybrid quantum-classical facial emotion recognition for smart-city surveillance. "
                              "Predictions are probabilistic and are not evidence of a person's mental state or intent.")
    app.state.settings = s
    app.state.db = Database(s.database_url)
    app.state.engine = InferenceEngine(s)
    app.state.anomaly = AnomalyDetector(AnomalyConfig(window_s=s.anomaly_window_s, min_duration_s=s.anomaly_min_duration_s,
                                                      min_faces=s.anomaly_min_faces, baseline_warmup=s.anomaly_baseline_warmup))
    app.state.alerts = AlertService(app.state.anomaly, s.anomaly_cooldown_s)
    app.state.ws = ws.ConnectionManager()

    app.add_middleware(CORSMiddleware, allow_origins=s.cors_list, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/api"):
            log.info("request", extra={"method": request.method, "path": request.url.path, "status": response.status_code,
                                       "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return response

    def err(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": message, "code": code})

    @app.exception_handler(NoFaceDetectedError)
    async def _no_face(_: Request, e: NoFaceDetectedError):
        return err(422, "no_face_detected", str(e))

    @app.exception_handler(InvalidImageError)
    async def _bad_image(_: Request, e: InvalidImageError):
        return err(400, "invalid_image", str(e))

    @app.exception_handler(ModelUnavailableError)
    async def _no_model(_: Request, e: ModelUnavailableError):
        return err(503, "model_unavailable", str(e))

    @app.exception_handler(QuantumCircuitError)
    async def _qerr(_: Request, e: QuantumCircuitError):
        return err(500, "quantum_circuit_error", "The quantum circuit could not be executed. Please try again.")

    @app.exception_handler(SQLAlchemyError)
    async def _dberr(_: Request, e: SQLAlchemyError):
        log.error("database error", extra={"error": str(e)[:200]})
        return err(503, "database_unavailable", "Database unavailable. Please try again shortly.")

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception):
        log.error("unhandled error", extra={"error": repr(e)[:300]})
        return err(500, "internal_error", "Unexpected server error.")

    for r in (auth.router, predict.router, cameras.router, alerts.router, info.router, ws.router):
        app.include_router(r)
    return app


app = create_app()
