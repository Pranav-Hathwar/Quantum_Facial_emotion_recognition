# QuantumVision API Reference

The QuantumVision backend is built with FastAPI. An interactive Swagger UI is available at `/docs` and ReDoc at `/redoc`.

---

## Authentication

All protected endpoints require a JWT Bearer token in the `Authorization` header:
```http
Authorization: Bearer <access_token>
```

### Roles & Permissions
* **`ADMIN`**: User management, camera management, alert operations, model evaluation queries.
* **`OPERATOR`**: Alert lifecycle review (acknowledge/resolve with operator notes), camera inspection, inference.
* **`VIEWER`**: Read-only access to dashboard, camera feeds, analytics, and metrics.

### Authentication Endpoints
* **`POST /api/auth/login`** (OAuth2 password request: `username=<email>&password=<password>`)
  * Returns: `{ "access_token": "...", "token_type": "bearer" }`
* **`GET /api/auth/me`**: Returns the profile and role of the authenticated user.
* **`GET /api/auth/users`** (`ADMIN` only): Returns the list of registered users.
* **`POST /api/auth/users`** (`ADMIN` only): Creates a new user (`name`, `email`, `password`, `role`).

---

## Health & System Information

* **`GET /api/health`** (Public)
  * Returns: `{ "status": "ok", "database": true, "models_available": ["hybrid", "classical", "control"], "auth_enabled": true, "store_images": false }`
* **`GET /api/emotions`** (Public)
  * Returns: `["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]`

---

## Inference & Prediction

* **`POST /api/predict?model=auto|hybrid|classical|control`**
  * Form-data: `file` (Image binary)
  * Returns: Detected face bounding box, predicted emotion, winning confidence, 7-class probability vector, and quantum Pauli-Z expectations (if hybrid).
* **`POST /api/predict/multiple?model=auto|hybrid|classical|control`**
  * Supports group/crowd images with multiple faces detected simultaneously.
  * Returns an array of face predictions with bounding boxes and per-face emotion breakdowns.
* **`POST /api/predict/frame`**
  * Internal/direct frame evaluation with camera metadata.

---

## Surveillance & WebSockets

* **`WS /ws/surveillance/{camera_id}?token=<jwt>`**
  * Client sends: Raw binary JPEG frames ($\le 640\text{ px}$ recommended) at configurable frequency (e.g. 2–5 FPS).
  * Server emits JSON event messages:
    * `frame_result`: Face count, inference processing latency, timestamp.
    * `emotion_prediction`: Detected bounding boxes, predicted emotions, confidence values.
    * `alert_created`: Emitted if the anomaly engine triggers a group-level distress alert.

---

## Alerts & Analytics

* **`GET /api/alerts`**
  * Filters: `camera_id`, `status` (`NEW`, `ACKNOWLEDGED`, `RESOLVED`), `severity` (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
* **`GET /api/alerts/{id}`**: Returns full alert detail including baseline comparison.
* **`POST /api/alerts/{id}/acknowledge`** (`OPERATOR` / `ADMIN`): Marks alert acknowledged with operator note.
* **`POST /api/alerts/{id}/resolve`** (`OPERATOR` / `ADMIN`): Resolves alert with closing remarks.
* **`GET /api/analytics`**: Aggregated emotion distribution timelines and incident metrics.

---

## Model & Quantum Circuit Introspection

* **`GET /api/model/info`**: Active model configuration, backend architecture, and parameter counts.
* **`GET /api/model/metrics`**: Measured FER2013 benchmark results from `comparison.json`.
* **`GET /api/quantum/circuit`**: Circuit architecture spec, qubit count, layer depth, and ASCII diagram.

---

## Error Codes & Responses

All API errors return consistent JSON:
```json
{
  "detail": "Descriptive message",
  "code": "error_identifier"
}
```

* **400 / 415**: Invalid or unreadable image file.
* **413**: File exceeds maximum upload size (10 MB).
* **422**: No face detected in the provided image.
* **503**: Serving model or database temporarily unavailable.

