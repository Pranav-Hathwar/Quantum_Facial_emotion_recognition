# QuantumVision

**Hybrid Quantum Deep Transfer Learning based Facial Emotion Recognition for Smart-City Surveillance**

An academic implementation of *Chapter 12 – A Conceptual Framework for Hybrid Quantum Deep Transfer Learning for
Emotion-Aware Surveillance in Next-Generation Smart Cities*.

> **Limitations.** Facial emotion predictions are probabilistic estimates of facial expression. They do not establish
> a person's mental state, intent or dangerousness. The quantum circuit runs on a **classical simulator**; **no quantum
> advantage is claimed** unless the project's own measured benchmarks show it. Alerts are statistical, group-level
> anomalies that require **human review**.

## Pipeline

```
image / camera -> face detection -> preprocessing -> ResNet50 (ImageNet transfer learning, 2048-d)
  -> dimensionality reduction (Linear 2048 -> n_qubits) -> angle encoding (R_y)
  -> variational quantum circuit (RX/RY/RZ + CNOT ring, PennyLane) -> Pauli-Z expectation values
  -> fusion (concat classical 256-d + quantum n-d) -> dense + softmax -> 7 emotions -> analytics / anomaly alerts
```

Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. Dataset: FER2013.

Models compared (identical frozen ResNet50 features, splits, optimiser and seeds):

| Model | Head |
|---|---|
| A  Classical baseline | Linear(2048,256) → ReLU → Dropout → Linear(256,7) |
| A′ Control | same, with a classical `tanh(Linear(n,n))` where the VQC would be (shows whether any gain is "quantum" or just "extra parameters") |
| B  Hybrid | reduce → R_y encoding → VQC → ⟨Z⟩ → concat with classical features → Linear(7) |

## Quick start

### 1. Environment (Windows PowerShell shown; Linux/macOS use `source .venv/bin/activate`)

```powershell
python -m venv .venv ; .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA build for an NVIDIA GPU
pip install -r requirements.txt
copy .env.example .env      # then edit JWT_SECRET / ADMIN_PASSWORD
```

### 2. Data (FER2013 as a zip of image folders, or fer2013.csv)

```powershell
python -m ml.preprocessing.zip_to_folders --zip archive.zip          # -> data/fer2013/{train,validation,test}/<emotion>/
# or: python -m ml.preprocessing.csv_to_folders --csv fer2013.csv
python scripts/fetch_face_model.py                                    # optional YuNet face model (else Haar cascade)
```

### 3. Train and compare (needs internet once for the ImageNet ResNet50 weights)

```powershell
python extract_features.py --device cuda                               # caches frozen ResNet50 features (once)
python run_experiments.py --seeds 42 43 44 --device cuda               # trains A, A', B for each seed + evaluates on test
python compare.py                                                      # writes ml/models/comparison.{json,md}
```

Fine-tuning (`layer4`/full network) is available with `--mode fine_tuning` in `train_classical.py` / `train_hybrid.py`.
Everything shown in the app's **Model Performance** page comes from `comparison.json`; if it does not exist the page says so.

### 4. Run the app

```powershell
uvicorn backend.app.main:app --reload --port 8000     # API + WebSocket, docs at /docs
cd frontend ; npm install ; npm run dev                # http://localhost:5173
```

Login: the `ADMIN_EMAIL` / `ADMIN_PASSWORD` from `.env` (defaults `admin@quantumvision.local` / `admin123` – change them).
Roles: `ADMIN` (everything, users, cameras), `OPERATOR` (acknowledge/resolve alerts), `VIEWER` (read-only).

### 5. Docker (PostgreSQL + backend + frontend)

```bash
docker compose up --build        # app at http://localhost:8080 ; mount trained models in ./ml/models
```

## Application pages

Dashboard · Live Surveillance (webcam or video file) · Image Analysis (multi-face) · Emotion Analytics ·
Alerts (NEW/ACKNOWLEDGED/RESOLVED, LOW→CRITICAL, operator notes) · Quantum Circuit · Model Performance · Cameras · Settings.

## Emotional-anomaly alerts

No single emotion is treated as dangerous. An alert is raised only for a **group-level, sustained** shift of
Fear/Surprise/Angry share relative to the camera's own rolling baseline (≥3 faces, ≥8 s inside a 20 s window,
≥60 % of frames elevated, 60 s cooldown). Details: [docs/architecture.md](docs/architecture.md).

## Testing

```bash
python -m pytest -q        # 59 tests (quantum circuit checked against an independent numpy state-vector, API, alerts, ...)
cd frontend && npm run build
```

## Honest status

**Measured results — FER2013 test split, 3 seeds (42, 43, 44), feature-extractor mode, trained on an RTX 4060 Laptop GPU on 2026-10-06.** Numbers come straight from `ml/models/comparison.json` (written by `compare.py`); they are not hand-entered and are reproducible with the commands above.

| Metric | Classical ResNet50 (A) | ResNet50 + classical layer (A′) | Hybrid ResNet50 + Quantum VQC (B) |
|---|---|---|---|
| Accuracy | 0.5493 ± 0.0104 | 0.5528 ± 0.0048 | 0.5547 ± 0.0055 |
| Macro F1 | 0.5191 ± 0.0147 | 0.5243 ± 0.0126 | 0.5233 ± 0.0077 |
| Training time (s) | 129 ± 32 | 181 ± 2 | 1837 ± 643 |
| Inference (ms/face, total) | 13.2 | 14.8 | 53.4 |
| Head parameters | 526,343 | 534,587 | 534,591 (24 quantum) |

* **No significant difference.** The hybrid's accuracy edge over the plain classical model is +0.0054, inside the seed-to-seed noise, and it is effectively tied with the fair control (A′, same wiring with a classical layer in place of the VQC). The exact McNemar test (classical vs hybrid) gives p = 0.52 / 0.001 / 0.96 across the three seeds — significant on only one, not consistently. There is **no quantum advantage and no speed-up**: the quantum circuit is a classical simulation (`default.qubit`), and the hybrid is ~14× slower to train and ~4× slower per face at inference.
* The large training-time spread for the hybrid (±643 s) reflects two mid-run stalls while the laptop was under load, not the circuit itself; inference times were measured in isolation and are stable.
* Predictions remain **probabilistic estimates of facial expression** and do not establish anyone's actual emotional state, intent, or dangerousness (see the disclaimer carried in every API response and in the UI).

## Documentation

[docs/architecture.md](docs/architecture.md) · [docs/ml_pipeline.md](docs/ml_pipeline.md) ·
[docs/quantum_pipeline.md](docs/quantum_pipeline.md) · [docs/api.md](docs/api.md) ·
[docs/database.md](docs/database.md) · [docs/migrations.md](docs/migrations.md) · [docs/deployment.md](docs/deployment.md)
