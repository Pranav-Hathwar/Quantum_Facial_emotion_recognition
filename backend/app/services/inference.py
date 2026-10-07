"""Inference engine: image -> faces -> ResNet50 -> (reduction -> quantum circuit -> fusion) -> 7 probabilities.

Only trained checkpoints produced by train_*.py are served. If none exist the API answers 503 "Model unavailable"
instead of inventing predictions.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from backend.app.config import AppSettings
from backend.app.schemas.api import BoundingBox, FaceResult, PipelineStage
from backend.app.utils.logging import get_logger
from ml.exceptions import NoFaceDetectedError, QuantumVisionError
from ml.labels import EMOTIONS
from ml.model_factory import load_serving_model
from ml.preprocessing.face_detector import FaceDetector
from ml.preprocessing.image_preprocessor import crop_face, preprocess_face

log = get_logger("inference")
MAX_FACES = 30


class ModelUnavailableError(QuantumVisionError):
    """No trained checkpoint is available for the requested model."""


class QuantumCircuitError(QuantumVisionError):
    """The quantum circuit simulation failed."""


@dataclass
class ModelBundle:
    kind: str
    run_name: str
    backbone: torch.nn.Module
    head: torch.nn.Module
    meta: dict

    @property
    def is_hybrid(self) -> bool:
        return self.kind == "hybrid"

    @property
    def is_smoke_test(self) -> bool:
        return self.meta.get("backbone_pretrained") is False


@dataclass
class Analysis:
    model: ModelBundle
    width: int
    height: int
    faces: list[FaceResult]
    pipeline: list[PipelineStage]
    timing_ms: dict
    warnings: list[str]


def _best_run(runs_dir: Path, kind: str) -> Path | None:
    best, best_f1 = None, -1.0
    for h in runs_dir.glob("*/history.json"):
        run = h.parent
        if not (run / "best.pt").is_file():
            continue
        try:
            hist = json.loads(h.read_text())
        except (OSError, ValueError):
            continue
        if hist.get("meta", {}).get("kind") != kind:
            continue
        f1 = hist.get("best_val_macro_f1", -1.0)
        if f1 > best_f1:
            best, best_f1 = run, f1
    return best


class InferenceEngine:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._bundles: dict[str, ModelBundle] = {}
        self._detector: FaceDetector | None = None
        self._lock = threading.Lock()

    # ---- model management ------------------------------------------------------------------------
    @property
    def runs_dir(self) -> Path:
        return self.settings.models_dir / "runs"

    def available_kinds(self) -> list[str]:
        return [k for k in ("hybrid", "classical", "control") if _best_run(self.runs_dir, k) is not None]

    def default_kind(self) -> str:
        pref = self.settings.serving_model
        kinds = self.available_kinds()
        if pref != "auto":
            return pref
        if not kinds:
            raise ModelUnavailableError("No trained model is available yet. Train one first (see README: train_classical.py / "
                                        "train_hybrid.py).")
        return kinds[0]

    def get_bundle(self, kind: str | None = None) -> ModelBundle:
        kind = kind or self.default_kind()
        if kind not in {"hybrid", "classical", "control"}:
            raise ModelUnavailableError(f"Unknown model '{kind}'.")
        with self._lock:
            if kind in self._bundles:
                return self._bundles[kind]
            run = _best_run(self.runs_dir, kind)
            if run is None:
                raise ModelUnavailableError(f"No trained '{kind}' model found under {self.runs_dir}.")
            try:
                t0 = time.perf_counter()
                backbone, head, meta = load_serving_model(run / "best.pt", self.settings.device,
                                                          self.settings.pretrained_backbone, self.settings.resnet_weights)
            except Exception as exc:  # noqa: BLE001
                log.error("model load failed", extra={"kind": kind, "error": str(exc)[:300]})
                raise ModelUnavailableError(f"The '{kind}' model could not be loaded: {exc}") from exc
            bundle = ModelBundle(kind, run.name, backbone, head, meta)
            self._bundles[kind] = bundle
            log.info("model loaded", extra={"kind": kind, "run": run.name,
                                            "seconds": round(time.perf_counter() - t0, 2)})
            return bundle

    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = FaceDetector(self.settings.face_detector)
            log.info("face detector ready", extra={"backend": self._detector.backend})
        return self._detector

    # ---- main entry point -------------------------------------------------------------------------
    def analyze(self, img_bgr: np.ndarray, model: str | None = None) -> Analysis:
        bundle = self.get_bundle(model)
        H, W = img_bgr.shape[:2]
        timing: dict[str, float] = {}
        warnings: list[str] = []

        with self._lock:
            t = time.perf_counter()
            faces = self.detector().detect(img_bgr)
            timing["face_detection_ms"] = (time.perf_counter() - t) * 1000
            if not faces:
                raise NoFaceDetectedError()
            if len(faces) > MAX_FACES:
                faces = sorted(faces, key=lambda f: f.width * f.height, reverse=True)[:MAX_FACES]
                faces.sort(key=lambda f: f.x)
                warnings.append(f"More than {MAX_FACES} faces found; only the {MAX_FACES} largest were analysed.")

            t = time.perf_counter()
            batch = np.stack([preprocess_face(crop_face(img_bgr, f.bbox), self.settings.image_size) for f in faces])
            x = torch.from_numpy(batch).to(self.settings.device)
            timing["preprocessing_ms"] = (time.perf_counter() - t) * 1000

            with torch.inference_mode():
                t = time.perf_counter()
                feats = bundle.backbone(x)
                timing["backbone_ms"] = (time.perf_counter() - t) * 1000
                inter: dict | None = None
                t = time.perf_counter()
                try:
                    if bundle.is_hybrid:
                        inter = bundle.head.forward_with_intermediates(feats, timing)
                        logits = inter["logits"]
                    else:
                        logits = bundle.head(feats)
                except Exception as exc:  # noqa: BLE001
                    log.error("quantum/classifier failure", extra={"error": str(exc)[:300]})
                    raise QuantumCircuitError(f"Model execution failed: {exc}") from exc
                timing["head_ms"] = (time.perf_counter() - t) * 1000
                probs = torch.softmax(logits, dim=1).cpu().numpy()

        results: list[FaceResult] = []
        for i, f in enumerate(faces):
            p = probs[i]
            top = int(p.argmax())
            results.append(FaceResult(
                face_id=f.face_id,
                bounding_box=BoundingBox(x=f.x, y=f.y, width=f.width, height=f.height),
                detection_confidence=round(float(f.confidence), 4),
                emotion=EMOTIONS[top], confidence=round(float(p[top]), 4),
                probabilities={e: round(float(v), 4) for e, v in zip(EMOTIONS, p)},
                quantum_features=[round(float(v), 5) for v in inter["quantum"][i].tolist()] if inter else None,
                quantum_angles=[round(float(v), 5) for v in inter["angles"][i].tolist()] if inter else None,
            ))
        timing["total_ms"] = sum(v for k, v in timing.items() if k in
                                 ("face_detection_ms", "preprocessing_ms", "backbone_ms", "head_ms"))
        log.info("prediction", extra={"model": bundle.kind, "faces": len(results),
                                      "total_ms": round(timing["total_ms"], 1)})
        if bundle.is_smoke_test:
            warnings.append("This model was trained on random (non-pretrained) backbone weights for a smoke test. "
                            "Its predictions are meaningless.")
        return Analysis(bundle, W, H, results, _pipeline(bundle, timing), {k: round(v, 2) for k, v in timing.items()}, warnings)


def _pipeline(bundle: ModelBundle, t: dict) -> list[PipelineStage]:
    hybrid = bundle.is_hybrid
    na = "not_applicable"
    q_note = "Encoding, VQC and measurement execute as one circuit on a classical simulator."
    stages = [
        PipelineStage(stage="face_detection", label="Face detected", domain="classical", status="ok", ms=t.get("face_detection_ms")),
        PipelineStage(stage="preprocessing", label="Preprocessing (crop, resize, normalise)", domain="classical", status="ok", ms=t.get("preprocessing_ms")),
        PipelineStage(stage="resnet50", label="ResNet50 feature extraction", domain="classical", status="ok", ms=t.get("backbone_ms")),
        PipelineStage(stage="reduction", label="Feature reduction", domain="classical",
                      status="ok" if hybrid else na, ms=t.get("feature_reduction_ms")),
        PipelineStage(stage="quantum_encoding", label="Quantum encoding (angle, R_y)", domain="quantum",
                      status="ok" if hybrid else na, note=q_note if hybrid else None),
        PipelineStage(stage="vqc", label="VQC execution", domain="quantum", status="ok" if hybrid else na,
                      ms=t.get("quantum_circuit_ms"), note=q_note if hybrid else None),
        PipelineStage(stage="measurement", label="Quantum measurement (Pauli-Z)", domain="quantum",
                      status="ok" if hybrid else na, note=q_note if hybrid else None),
        PipelineStage(stage="classification", label="Fusion + 7-class classification", domain="classical", status="ok",
                      ms=t.get("fusion_classification_ms", t.get("head_ms"))),
    ]
    return stages
