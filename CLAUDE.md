# QuantumVision – context for Claude Code

Academic app implementing ONLY Chapter 12 of the user's book (hybrid quantum deep transfer learning for emotion-aware
surveillance). Pipeline: face detection -> ResNet50 (frozen, cached features) -> Linear(2048->n_qubits) -> R_y angle
encoding -> PennyLane VQC (RX/RY/RZ + CNOT ring) -> Pauli-Z -> concat with classical features -> softmax over 7 emotions.

## Rules
- Never fabricate metrics or claim quantum advantage; the quantum circuit is a classical simulation.
- Alerts are group-level, sustained anomalies for human review; never "angry = dangerous".
- Keep the "predictions are probabilistic" disclaimer in the UI.

## State
- Done: ML pipeline, evaluation/compare tooling, FastAPI backend (JWT roles, WebSocket, anomaly engine), React frontend (all pages), docs/, Docker. 59 tests pass (`python -m pytest -q`); `cd frontend && npm run build` passes.
- NOT done: real training. No results exist yet. Run `run_all.ps1 -Zip <archive.zip>` on the NVIDIA GPU laptop (Windows PowerShell).
- Next: train/evaluate (Phases 4-7), check ml/models/comparison.json shows in Model Performance page, UI polish, Alembic migrations (tables currently via create_all), final report.
- Docs: README.md and docs/*.md. Plan/progress notes are in the Claude project docs.

## Task checklist for Claude Code (work in order; tick off and verify each)
1. Environment: create .venv, install CUDA torch (`--index-url https://download.pytorch.org/whl/cu121`), `pip install -r requirements.txt`; confirm `torch.cuda.is_available()`.
2. Data: `python -m ml.preprocessing.zip_to_folders --zip <archive.zip>`; confirm counts ~28.7k/3.6k/3.6k.
3. `python -m pytest -q` must pass before training.
4. `python extract_features.py --device cuda` (needs internet once for ResNet50 weights).
5. `python run_experiments.py --seeds 42 43 44 --device cuda` (if the hybrid is too slow, reduce epochs and say so; do not drop seeds silently).
6. `python compare.py`; read ml/models/comparison.md and report the REAL numbers, including if hybrid is worse or equal.
7. Run backend + frontend (`uvicorn backend.app.main:app --port 8000`, `cd frontend && npm install && npm run dev`), log in, open Model Performance and verify it shows the measured results; test image upload with real faces and live webcam.
8. Fix any bug found; keep tests green; update README "Honest status" with the measured results (never invent numbers).
9. Optional: `--mode fine_tuning` experiment, Alembic migrations, report-ready figures.
Definition of done: comparison.json exists from real runs, app shows it, tests pass, README updated with true results.
