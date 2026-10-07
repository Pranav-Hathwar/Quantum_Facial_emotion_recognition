"""Backend settings (env vars / .env). ML settings live in ml/config.py and are re-used here."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    app_name: str = "QuantumVision"
    environment: str = "development"
    # SQLite for local dev; docker-compose / production sets a postgresql+psycopg:// URL.
    database_url: str = f"sqlite:///{ROOT / 'quantumvision.db'}"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    jwt_secret: str = "change-me-in-production-use-a-long-random-string-0123456789"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 12
    admin_email: str = "admin@quantumvision.local"
    admin_password: str = "admin123"        # demo default - change in .env
    auth_disabled: bool = False             # only for quick local demos

    # model serving
    model_path: str = "ml/models"
    serving_model: str = "auto"             # "auto" | "hybrid" | "classical" | "control"
    pretrained_backbone: bool = True        # tests set False (no weight download)
    resnet_weights: str = "v1"
    device: str = "cpu"
    image_size: int = 224
    face_detector: str = "auto"
    max_upload_mb: int = 10
    store_images: bool = False              # privacy default: never store raw faces
    process_interval_ms: int = 500

    # emotion-anomaly engine
    anomaly_window_s: float = 20.0
    anomaly_min_duration_s: float = 8.0
    anomaly_min_faces: float = 3.0
    anomaly_baseline_warmup: int = 60       # face observations before a baseline is trusted
    anomaly_cooldown_s: float = 60.0

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def models_dir(self) -> Path:
        p = Path(self.model_path)
        return p if p.is_absolute() else ROOT / p


@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings()
