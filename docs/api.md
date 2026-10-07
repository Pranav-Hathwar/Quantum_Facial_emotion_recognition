# API (interactive docs at `/docs`)

Auth: `POST /api/auth/login` (form: username=email, password) → JWT; send `Authorization: Bearer <token>`.

| Method & path | Role | Purpose |
|---|---|---|
| GET `/api/health` | public | status, DB, models available |
| GET `/api/emotions` | public | the 7 classes |
| GET `/api/auth/me`, `/api/auth/users` · POST `/api/auth/users` | user / ADMIN | profile, user admin |
| POST `/api/predict`, `/api/predict/multiple`, `/api/predict/frame` | user | image → faces + 7 probabilities (+ quantum measurements). `?model=hybrid|classical|control` |
| GET `/api/analytics?camera_id&minutes&bucket_seconds` | user | distribution, timeline, totals |
| GET `/api/alerts` · GET `/api/alerts/{id}` | user | list / detail (filters: status, severity, camera_id) |
| POST `/api/alerts/{id}/acknowledge`, `/resolve` | OPERATOR | workflow with note |
| GET/POST `/api/cameras` · PATCH `/api/cameras/{id}` | user / ADMIN | camera registry |
| GET `/api/model/info`, `/api/model/metrics`, `/api/quantum/circuit` | user | model details, measured comparison, circuit spec |
| WS `/ws/surveillance/{camera_id}?token=` | user | binary JPEG frames in; `frame_result`, `emotion_prediction`, `alert_created` out |

Errors are JSON `{detail, code}`: invalid image (400/415), no face (422), file too large (413), model/DB unavailable (503).
