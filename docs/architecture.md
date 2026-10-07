# Architecture

```
React (Vite, Tailwind, Recharts) --HTTP/WebSocket--> FastAPI --> ml/ (ResNet50, PennyLane VQC, face detector)
                                                        \--> SQLAlchemy --> PostgreSQL (SQLite for dev/tests)
```

* `ml/` – shared ML code: `preprocessing/`, `classical/`, `quantum/`, `training/`, `evaluation/`, `model_factory.py`.
  The backend imports the same preprocessing and model code that training uses, so train/serve behaviour cannot drift.
* `backend/app/` – `api/` (routers), `services/` (inference, anomaly engine, alerts, store, surveillance), `models/` (ORM),
  `schemas/` (Pydantic), `database/`, `utils/`.
* `frontend/src/` – `pages/`, `components/`, `hooks/` (auth, polling, surveillance socket), `services/api.js`.

## Chapter 12 vs engineering decisions

| From Chapter 12 | Our engineering decisions |
|---|---|
| FER2013, 7 emotions, ResNet50 transfer learning, angle encoding, VQC, Pauli-Z measurement, fusion of classical+quantum features, hybrid training, surveillance use case | PyTorch/PennyLane/FastAPI/React/PostgreSQL, classical control model A′, cached-feature protocol, multi-seed + McNemar evaluation, JWT roles, privacy default, **group-level anomaly alerts instead of per-person "aggressive" flags** |

## Live surveillance flow

Browser camera or video file → JPEG frame (≤640 px) → WebSocket binary message at the chosen analysis rate →
server detects every face, runs each through the model → `frame_result` / `emotion_prediction` events; the anomaly
engine consumes each frame's emotion distribution; `alert_created` events are pushed. Frames are dropped, not queued,
when the server is behind. Raw frames are not stored unless `STORE_IMAGES=true`.

## Anomaly engine (`backend/app/services/anomaly.py`)

Per camera: a decaying baseline of the Fear+Surprise+Angry share; a 20 s sliding window; an alert needs
≥3 faces in frame, ≥8 s of sustained elevation, ≥60 % of window frames elevated. Score = (distress_now − baseline)/(1 − baseline),
mapped to LOW/MEDIUM/HIGH/CRITICAL. 60 s cooldown per camera; the baseline is frozen during an anomaly; no alert
fires until the baseline has seen enough observations. Alerts are **for human review** and carry operator notes.
