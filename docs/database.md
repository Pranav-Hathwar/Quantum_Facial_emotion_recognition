# Database

Tables (SQLAlchemy 2; created at startup with `create_all`, so there are no Alembic migrations yet):
`users` (email, bcrypt hash, role), `cameras`, `emotion_predictions` (camera, model, label, 7 probabilities, time),
`alerts` (camera, severity, status, anomaly score, baseline/current distributions, operator notes, timestamps),
`model_runs` (name, version, accuracy, precision, recall, F1, inference time).
Raw images are **not** stored unless `STORE_IMAGES=true`. SQLite is used for dev/tests, PostgreSQL via `DATABASE_URL` in Docker.
