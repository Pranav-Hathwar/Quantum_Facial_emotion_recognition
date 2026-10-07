"""ORM tables: users, cameras, emotion_predictions, alerts, model_runs."""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, enum.Enum):
    ADMIN = "ADMIN"
    OPERATOR = "OPERATOR"
    VIEWER = "VIEWER"


class AlertStatus(str, enum.Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class Severity(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default=Role.VIEWER.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)      # e.g. CAM-001
    name: Mapped[str] = mapped_column(String(120))
    location: Mapped[str] = mapped_column(String(200), default="")
    source: Mapped[str] = mapped_column(String(500), default="webcam")  # webcam | video file | rtsp url (future)
    status: Mapped[str] = mapped_column(String(20), default="OFFLINE")
    last_active: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EmotionPrediction(Base):
    __tablename__ = "emotion_predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str | None] = mapped_column(ForeignKey("cameras.id"), nullable=True, index=True)
    analysis_id: Mapped[str] = mapped_column(String(36), index=True, default="")  # groups the faces of one image/frame
    face_id: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String(40), default="")
    emotion: Mapped[str] = mapped_column(String(20), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(20), default="live")     # live | upload
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    alert_type: Mapped[str] = mapped_column(String(40), default="EMOTIONAL_ANOMALY")
    severity: Mapped[str] = mapped_column(String(10))
    anomaly_score: Mapped[float] = mapped_column(Float)
    emotion_distribution: Mapped[dict] = mapped_column(JSON)
    baseline_distribution: Mapped[dict] = mapped_column(JSON, default=dict)
    face_count: Mapped[float] = mapped_column(Float, default=0)
    duration_s: Mapped[float] = mapped_column(Float, default=0)
    status: Mapped[str] = mapped_column(String(20), default=AlertStatus.NEW.value, index=True)
    operator_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    handled_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelRun(Base):
    __tablename__ = "model_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(80))
    model_version: Mapped[str] = mapped_column(String(80), default="")
    accuracy: Mapped[float] = mapped_column(Float)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    f1_score: Mapped[float] = mapped_column(Float)
    inference_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
