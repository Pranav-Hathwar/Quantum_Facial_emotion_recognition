# Database Architecture

The application uses SQLAlchemy 2 with support for SQLite (local development and tests) and PostgreSQL (Docker and production environments via `psycopg`).

## Tables & Models

1. **`users`**:
   - `id`: Primary key.
   - `name`: Display name.
   - `email`: Unique email identifier (lowercased).
   - `password_hash`: Bcrypt-hashed password.
   - `role`: Role-based access control (`ADMIN`, `OPERATOR`, `VIEWER`).
   - `created_at`: Account creation timestamp.

2. **`cameras`**:
   - `id`: String identifier (e.g., `CAM-001`).
   - `name`: Human-readable label.
   - `location`: Physical site description.
   - `source`: Input source (`webcam`, `video-file`, or stream URL).
   - `is_active`: Operational status flag.
   - `created_at`: Registration timestamp.

3. **`emotion_predictions`**:
   - `id`: Primary key.
   - `camera_id`: Foreign key reference to `cameras.id`.
   - `timestamp`: Detection timestamp.
   - `face_index`: Face identifier in multi-face frames.
   - `bounding_box`: JSON bounding box coordinates `[x, y, w, h]`.
   - `predicted_emotion`: Argmax predicted class (one of 7 FER emotions).
   - `confidence`: Confidence score for winning class.
   - `probabilities`: JSON array with all 7 softmax probabilities.
   - `model_used`: Name of model serving the inference (`hybrid`, `classical`, `control`).
   - `quantum_measurements`: JSON array of Pauli-Z expectation values (if hybrid).
   - `image_path`: Filepath if image persistence is explicitly enabled (`STORE_IMAGES=true`).

4. **`alerts`**:
   - `id`: Primary key.
   - `camera_id`: Foreign key reference to `cameras.id`.
   - `created_at`: Alert trigger timestamp.
   - `severity`: Alert priority (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
   - `status`: Incident lifecycle state (`NEW`, `ACKNOWLEDGED`, `RESOLVED`).
   - `anomaly_score`: Normalized anomaly metric (0.0 to 1.0).
   - `fear_pct`, `surprise_pct`, `angry_pct`: Group-level emotion share in window.
   - `baseline_fear_pct`, `baseline_surprise_pct`, `baseline_angry_pct`: Rolling camera baseline at trigger time.
   - `face_count`: Number of faces detected during anomaly window.
   - `sustained_seconds`: Duration the condition persisted.
   - `notes`: Human operator review notes.
   - `resolved_by`: User ID of operator resolving the alert.
   - `resolved_at`: Resolution timestamp.

5. **`model_runs`**:
   - `id`: Primary key.
   - `run_name`: Unique training run identifier.
   - `model_kind`: Architecture (`classical`, `control`, `hybrid`).
   - `mode`: Training mode (`feature_extractor` or `fine_tuning`).
   - `seed`: Random seed.
   - `accuracy`, `macro_f1`, `weighted_f1`: Measured benchmark scores.
   - `inference_ms`: Total inference latency per face (milliseconds).
   - `is_best`: Boolean flag for the selected default serving weights.

## Schema Creation & Migrations

* **Development & Automated Tests**: Initialized via `Database.create_all()` during startup in `backend/app/main.py`.
* **Production Migrations**: Managed via Alembic (`alembic upgrade head`). See [docs/migrations.md](migrations.md) for CLI commands and workflow details.

## Privacy & Storage Policies

* By default, **raw facial images are never stored** (`STORE_IMAGES=false`). Only statistical metadata, predicted emotion distributions, and group-level anomaly metrics are saved to the database.

