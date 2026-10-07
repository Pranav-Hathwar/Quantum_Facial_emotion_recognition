# QuantumVision

**Hybrid Quantum Deep Transfer Learning for Facial Emotion Recognition in Smart-City Surveillance**

An academic reference implementation of *Chapter 12 – A Conceptual Framework for Hybrid Quantum Deep Transfer Learning for Emotion-Aware Surveillance in Next-Generation Smart Cities*.

> [!NOTE]
> **Probabilistic Predictions & Ethical Safeguards:** Facial emotion predictions are probabilistic estimates of facial expression. They do not establish an individual's internal mental state, intent, or dangerousness. The quantum circuit runs on a **classical state-vector simulator**; **no quantum advantage or physical speed-up is claimed**. Alerts are statistical, group-level anomalies intended strictly for **human operator review**.

---

## Architecture & Pipeline

```text
Input Video/Image ──► Face Detection (YuNet / Haar) ──► Crop & Resize (224x224)
                           │
                           ▼
              ResNet50 Transfer Learning (ImageNet, 2048-d)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Classical Feature (256-d)    Linear Reduction (2048 -> n_qubits)
             │                           │
             │                   Angle Encoding (R_y)
             │                           │
             │                   PennyLane Variational Quantum Circuit (VQC)
             │                   [RX, RY, RZ rotations + CNOT ring topology]
             │                           │
             │                   Pauli-Z Expectation Measurement <Z>
             └─────────────┬─────────────┘
                           ▼
            Concatenation & Fusion (256 + n_qubits)
                           │
                           ▼
          Dense Classifier + Softmax (7 Emotions)
     [Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]
                           │
                           ▼
        Surveillance Stream / Anomaly Alert Engine
```

### Models Benchmarked

All models use identical frozen ResNet50 representations, dataset splits, optimizer settings, and random seeds:

| Model | Head Architecture | Purpose |
|---|---|---|
| **A (Classical baseline)** | `Linear(2048, 256) -> ReLU -> Dropout -> Linear(256, 7)` | Standard deep transfer learning benchmark |
| **A′ (Control model)** | Identical to B, replacing the VQC with `tanh(Linear(n, n))` | Isolates whether changes stem from parameter count vs quantum circuit |
| **B (Hybrid VQC)** | `Linear(2048, 4) -> R_y -> VQC(2 layers) -> <Z> -> Concat -> Linear(7)` | Quantum-classical hybrid architecture |

---

## Project Structure

```text
Quantum_Facial_emotion_recognition/
├── backend/                  # FastAPI backend server
│   ├── alembic/              # Database schema migrations
│   └── app/                  # API routers, services, models, and WebSocket engine
├── data/                     # Dataset directories (FER2013)
├── deploy/                   # Production Dockerfiles and Nginx reverse proxy configuration
├── docs/                     # Detailed architectural, API, and pipeline documentation
│   ├── api.md
│   ├── architecture.md
│   ├── database.md
│   ├── deployment.md
│   ├── migrations.md
│   ├── ml_pipeline.md
│   └── quantum_pipeline.md
├── frontend/                 # React 18 single-page application (Vite + Tailwind CSS)
│   ├── src/
│   │   ├── components/       # UI charts, overlays, layout, video feed
│   │   ├── hooks/            # Authentication, polling, WebSocket hooks
│   │   ├── pages/            # Dashboard, Live Surveillance, Image Analysis, etc.
│   │   └── services/         # API client
│   └── package.json
├── logs/                     # Application and experiment run logs (.gitkeep)
├── ml/                       # Core ML, PennyLane VQC, and preprocessing routines
│   ├── classical/            # ResNet50 backbone and classical heads
│   ├── evaluation/           # Metrics calculation and statistical comparison (McNemar)
│   ├── preprocessing/        # Face detection (YuNet/Haar) and FER2013 parsing
│   └── quantum/              # VQC circuit definitions, angle encoding, and measurements
├── tests/                    # Comprehensive unit and integration test suite (Pytest)
├── compare.py                # Statistical model comparison generator
├── extract_features.py       # ResNet50 feature cache extractor
├── requirements.txt          # Python dependencies
├── run_experiments.py        # Reproducible multi-seed training suite
├── train_classical.py        # Classical model training script
└── train_hybrid.py           # Hybrid VQC training script
```

---

## Quick Start

### 1. Prerequisites

* **Python 3.10+** (Python 3.12 recommended)
* **Node.js 18+** & npm
* *(Optional)* NVIDIA GPU with CUDA 12.x for accelerated training

### 2. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/Pranav-Hathwar/Quantum_Facial_emotion_recognition.git
cd Quantum_Facial_emotion_recognition

# (Optional) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Linux/macOS: source .venv/bin/activate

# Install PyTorch (CUDA build recommended if GPU is available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt

# Create environment configuration
copy .env.example .env   # On Linux/macOS: cp .env.example .env
```

### 3. Running the Application

#### Start the Backend (FastAPI)
```powershell
# From repository root:
py -3.12 -m uvicorn backend.app.main:app --port 8000 --host 127.0.0.1 --reload
```
* **API Server:** http://127.0.0.1:8000
* **Interactive OpenAPI Docs:** http://127.0.0.1:8000/docs
* **Health Check:** http://127.0.0.1:8000/api/health

#### Start the Frontend (Vite + React)
```powershell
cd frontend
npm install
npm run dev
```
* **Web UI:** http://localhost:5173

#### Default Credentials
* **Email:** `admin@quantumvision.local`
* **Password:** `admin123` *(configurable in `.env`)*

---

## Docker Deployment

To launch the full stack with PostgreSQL, FastAPI backend, and Nginx-served frontend:

```bash
docker compose up --build
```
* Access the app at **http://localhost:8080**.

---

## Data Preparation & Training Pipeline

### 1. Dataset Extraction
Place your FER2013 archive zip in the root directory:
```powershell
python -m ml.preprocessing.zip_to_folders --zip archive.zip
```

### 2. Cache Frozen Features
```powershell
python extract_features.py --device cuda
```

### 3. Run Multi-Seed Experiment
```powershell
python run_experiments.py --seeds 42 43 44 --device cuda
python compare.py
```
This generates `ml/models/comparison.json` and `ml/models/comparison.md` containing aggregate metrics and McNemar statistical tests.

---

## Empirical Benchmark Results

**FER2013 test split, 3 seeds (42, 43, 44), feature-extractor mode, trained on an NVIDIA RTX 4060 Laptop GPU:**

| Metric | Classical ResNet50 (A) | Fair Control (A′) | Hybrid ResNet50 + VQC (B) |
|---|---|---|---|
| **Accuracy** | 0.5493 ± 0.0104 | 0.5528 ± 0.0048 | **0.5547 ± 0.0055** |
| **Macro F1** | 0.5191 ± 0.0147 | **0.5243 ± 0.0126** | 0.5233 ± 0.0077 |
| **Training Time (s)** | **129 ± 32** | 181 ± 2 | 1837 ± 643 |
| **Inference (ms/face)** | **13.2 ms** | 14.8 ms | 53.4 ms |
| **Head Parameters** | 526,343 | 534,587 | 534,591 *(24 quantum)* |

* **Empirical Analysis**: The hybrid model's accuracy difference (+0.0054 over baseline A) falls within seed-to-seed variance, and ties with the classical control model A′. McNemar statistical testing ($p = 0.52 / 0.001 / 0.96$) demonstrates no consistent statistical advantage.
* **Simulation Overhead**: Because PennyLane simulates the quantum state vector classically, training latency is $\sim 14\times$ slower and inference is $\sim 4\times$ slower per face.

---

## Testing & Verification

Run the full automated test suite:
```powershell
py -3.12 -m pytest -q
```
*(Tests include quantum state-vector simulation against NumPy, API endpoints, JWT security, face detectors, and anomaly alert engines.)*

Validate frontend production build:
```powershell
cd frontend
npm run build
```

---

## Documentation

* [Architecture Overview](docs/architecture.md)
* [API Reference & WebSockets](docs/api.md)
* [Database Schema & Storage Policy](docs/database.md)
* [Database Migrations (Alembic)](docs/migrations.md)
* [ML Training & Evaluation Pipeline](docs/ml_pipeline.md)
* [PennyLane Quantum Circuit Specification](docs/quantum_pipeline.md)
* [Deployment Guide](docs/deployment.md)
