"""Group-level, sustained emotional-anomaly detection (our engineering addition - NOT in Chapter 12).

Chapter 12 says alerts could fire for "aggressive behaviour". We deliberately do NOT do that: a single face showing
Angry (or Fear, Surprise) means nothing on its own, and facial expression is not evidence of intent or danger.
Instead we flag a *statistical deviation from the camera's own baseline* that is
    * group-level   - averaged over many faces (>= min_faces per frame),
    * sustained     - lasts at least min_duration_s and most frames in the window are elevated,
    * baseline-relative - measured against what this camera normally sees (decaying history that excludes anomalies).
Every alert is only a prompt for HUMAN REVIEW.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ml.labels import EMOTIONS, EMOTION_TO_INDEX

DISTRESS_EMOTIONS = ("fear", "surprise", "angry")
_DISTRESS_IDX = [EMOTION_TO_INDEX[e] for e in DISTRESS_EMOTIONS]
SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

# What an everyday public space is assumed to look like until the camera has its own history.
_PRIOR = np.array([0.04, 0.01, 0.04, 0.25, 0.08, 0.04, 0.54])  # angry..neutral, sums to 1
_PRIOR_STRENGTH = 20.0   # pseudo-observations backing the prior
_BASELINE_HALF_LIFE_S = 600.0

DISCLAIMER = ("Statistical emotional-expression anomaly for HUMAN REVIEW only. Facial emotion predictions are "
              "probabilistic and do not establish anyone's mental state, intent or dangerousness.")


@dataclass
class AnomalyConfig:
    window_s: float = 20.0
    min_duration_s: float = 8.0
    min_faces: float = 3.0          # mean faces per frame across the window
    baseline_warmup: int = 60       # face observations before the camera baseline is trusted
    elevated_margin: float = 0.15   # a frame is "elevated" if its distress share > baseline + margin
    sustained_fraction: float = 0.6
    min_score: float = 0.2


@dataclass
class AnomalyResult:
    is_anomalous: bool
    severity: str | None
    score: float
    window_distribution: dict
    baseline_distribution: dict
    mean_faces: float
    duration_s: float
    sustained_fraction: float
    baseline_trusted: bool
    distress_share: float
    baseline_distress_share: float
    reason: str = ""


@dataclass
class _CameraState:
    frames: deque = field(default_factory=deque)      # (ts, counts[7], n_faces)
    base_counts: np.ndarray = field(default_factory=lambda: np.zeros(len(EMOTIONS)))
    base_obs: float = 0.0
    last_ts: float | None = None


def _dist(counts: np.ndarray) -> np.ndarray:
    s = counts.sum()
    return counts / s if s > 0 else np.zeros_like(counts)


def _as_dict(vec: np.ndarray) -> dict:
    return {e: round(float(v), 4) for e, v in zip(EMOTIONS, vec)}


class AnomalyDetector:
    def __init__(self, cfg: AnomalyConfig | None = None):
        self.cfg = cfg or AnomalyConfig()
        self._cams: dict[str, _CameraState] = {}

    def _state(self, camera_id: str) -> _CameraState:
        return self._cams.setdefault(camera_id, _CameraState())

    def baseline(self, camera_id: str) -> np.ndarray:
        st = self._state(camera_id)
        counts = st.base_counts + _PRIOR * _PRIOR_STRENGTH
        return _dist(counts)

    def reset(self, camera_id: str) -> None:
        self._cams.pop(camera_id, None)

    def observe(self, camera_id: str, ts: float, emotions: list[str]) -> AnomalyResult:
        """Feed one processed frame (the emotion label of every face in it) and evaluate the sliding window."""
        cfg, st = self.cfg, self._state(camera_id)
        counts = np.zeros(len(EMOTIONS))
        for e in emotions:
            counts[EMOTION_TO_INDEX[e]] += 1
        st.frames.append((ts, counts, len(emotions)))
        while st.frames and ts - st.frames[0][0] > cfg.window_s:
            st.frames.popleft()

        result = self._evaluate(camera_id, st)
        # only learn the baseline from "normal" time, otherwise the anomaly would be absorbed into it
        if not result.is_anomalous and not self._window_elevated(st, result):
            if st.last_ts is not None:
                decay = 0.5 ** (max(ts - st.last_ts, 0.0) / _BASELINE_HALF_LIFE_S)
                st.base_counts *= decay
                st.base_obs *= decay
            st.base_counts += counts
            st.base_obs += counts.sum()
        st.last_ts = ts
        return result

    # ------------------------------------------------------------------
    def _window_elevated(self, st: _CameraState, r: AnomalyResult) -> bool:
        return r.baseline_trusted and r.score >= self.cfg.min_score and r.mean_faces >= self.cfg.min_faces

    def _evaluate(self, camera_id: str, st: _CameraState) -> AnomalyResult:
        cfg = self.cfg
        frames = list(st.frames)
        total = np.sum([c for _, c, _ in frames], axis=0) if frames else np.zeros(len(EMOTIONS))
        n_obs = float(total.sum())
        window = _dist(total)
        base = self.baseline(camera_id)
        trusted = bool(st.base_obs >= cfg.baseline_warmup)
        duration = (frames[-1][0] - frames[0][0]) if len(frames) > 1 else 0.0
        mean_faces = float(np.mean([n for _, _, n in frames])) if frames else 0.0

        distress_now = float(window[_DISTRESS_IDX].sum()) if n_obs else 0.0
        distress_base = float(base[_DISTRESS_IDX].sum())
        score = max(0.0, distress_now - distress_base) / max(1.0 - distress_base, 1e-6)
        score = float(min(score, 1.0))

        elevated = [c[_DISTRESS_IDX].sum() / n > distress_base + cfg.elevated_margin
                    for _, c, n in frames if n > 0]
        sustained = float(np.mean(elevated)) if elevated else 0.0

        reason = ""
        ok = True
        if not trusted:
            ok, reason = False, "baseline still warming up"
        elif mean_faces < cfg.min_faces:
            ok, reason = False, f"fewer than {cfg.min_faces:g} faces per frame on average (group-level signal required)"
        elif duration < cfg.min_duration_s:
            ok, reason = False, "change not sustained long enough"
        elif sustained < cfg.sustained_fraction:
            ok, reason = False, "elevated in too few frames of the window"
        elif score < cfg.min_score:
            ok, reason = False, "deviation from baseline too small"

        severity = None
        if ok:
            if score >= 0.8 and mean_faces >= 8 and duration >= 0.75 * cfg.window_s:
                severity = "CRITICAL"
            elif score >= 0.6 and mean_faces >= 5:
                severity = "HIGH"
            elif score >= 0.4:
                severity = "MEDIUM"
            else:
                severity = "LOW"
        return AnomalyResult(ok, severity, score, _as_dict(window), _as_dict(base), mean_faces, float(duration),
                             sustained, trusted, distress_now, distress_base, reason)
