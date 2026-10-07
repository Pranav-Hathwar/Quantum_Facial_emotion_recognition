"""Central configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    dataset_path: Path
    model_path: Path
    image_size: int
    resnet_mode: str          # "feature_extractor" | "fine_tuning"
    n_qubits: int             # QUANTUM_FEATURES == N_QUBITS (one feature per qubit)
    n_q_layers: int
    quantum_backend: str
    learning_rate: float
    batch_size: int
    device: str
    store_images: bool        # privacy: raw faces are NOT stored unless explicitly enabled
    process_interval_ms: int
    face_detector: str        # "auto" | "yunet" | "haar"
    database_url: str

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        n_qubits = int(env("N_QUBITS", env("QUANTUM_FEATURES", "4")))
        mode = env("RESNET_MODE", "feature_extractor")
        if mode not in {"feature_extractor", "fine_tuning"}:
            raise ValueError("RESNET_MODE must be 'feature_extractor' or 'fine_tuning'")
        return cls(
            dataset_path=_path(env("DATASET_PATH", "data/fer2013")),
            model_path=_path(env("MODEL_PATH", "ml/models")),
            image_size=int(env("IMAGE_SIZE", "224")),
            resnet_mode=mode,
            n_qubits=n_qubits,
            n_q_layers=int(env("N_Q_LAYERS", "2")),
            quantum_backend=env("QUANTUM_BACKEND", "default.qubit"),
            learning_rate=float(env("LEARNING_RATE", "0.001")),
            batch_size=int(env("BATCH_SIZE", "32")),
            device=env("DEVICE", "cpu"),
            store_images=_bool(env("STORE_IMAGES", "false")),
            process_interval_ms=int(env("PROCESS_INTERVAL_MS", "500")),
            face_detector=env("FACE_DETECTOR", "auto"),
            database_url=env("DATABASE_URL", "postgresql+psycopg://quantumvision:quantumvision@localhost:5432/quantumvision"),
        )


def get_settings() -> Settings:
    return Settings.from_env()
