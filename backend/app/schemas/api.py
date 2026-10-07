"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = ("Facial emotion predictions are probabilistic estimates of facial expression. They do not establish a "
              "person's actual emotional or mental state, intent, or dangerousness, and can be affected by lighting, "
              "occlusion, camera angle, image quality, pose, dataset bias and individual differences.")


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class FaceResult(BaseModel):
    face_id: int
    bounding_box: BoundingBox
    detection_confidence: float
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    quantum_features: list[float] | None = None   # Pauli-Z expectation values (hybrid model only)
    quantum_angles: list[float] | None = None     # angle-encoded rotation angles (hybrid model only)


class PipelineStage(BaseModel):
    stage: str
    label: str
    domain: str                     # "classical" | "quantum"
    status: str                     # "ok" | "not_applicable"
    ms: float | None = None
    note: str | None = None


class PredictResponse(BaseModel):
    model: str
    model_kind: str
    image_width: int
    image_height: int
    face_count: int
    faces: list[FaceResult]
    prediction: str | None = None                 # primary (largest) face
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    pipeline: list[PipelineStage]
    timing_ms: dict[str, float]
    disclaimer: str = DISCLAIMER
    warnings: list[str] = Field(default_factory=list)
    alert: "AlertOut | None" = None
    camera_id: str | None = None


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    camera_id: str
    alert_type: str
    severity: str
    anomaly_score: float
    emotion_distribution: dict
    baseline_distribution: dict
    face_count: float
    duration_s: float
    status: str
    operator_action: str | None = None
    handled_by: str | None = None
    timestamp: datetime
    updated_at: datetime
    disclaimer: str = ("Statistical emotional-expression anomaly for human review only - not evidence of intent or "
                       "dangerousness.")


class AlertAction(BaseModel):
    note: str | None = Field(default=None, max_length=1000)


class CameraCreate(BaseModel):
    id: str | None = Field(default=None, max_length=32)
    name: str = Field(min_length=1, max_length=120)
    location: str = Field(default="", max_length=200)
    source: str = Field(default="webcam", max_length=500)


class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    location: str | None = Field(default=None, max_length=200)
    source: str | None = Field(default=None, max_length=500)


class CameraOut(BaseModel):
    id: str
    name: str
    location: str
    source: str
    status: str
    last_active: datetime | None
    current_emotion: str | None = None
    face_count: int = 0
    alert_state: str = "No Alert"        # "No Alert" | "<SEVERITY> ALERT"


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=128)
    role: str = "VIEWER"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: str
    role: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


PredictResponse.model_rebuild()
