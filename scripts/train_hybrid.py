"""Model B - hybrid ResNet50 + Quantum VQC (and Model A' control with --kind control).
    python train_hybrid.py
    python train_hybrid.py --kind control
    python train_hybrid.py --qubits 8 --q-layers 3
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import get_settings
from ml.training.run import run_train_finetune, run_train_head
from ml.training.trainer import TrainConfig

if __name__ == "__main__":
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["hybrid", "control"], default="hybrid")
    ap.add_argument("--mode", choices=["feature_extractor", "fine_tuning"], default=s.resnet_mode)
    ap.add_argument("--data", type=Path, default=s.dataset_path)
    ap.add_argument("--cache", type=Path, default=s.model_path / "features")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=s.learning_rate)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=s.device)
    ap.add_argument("--qubits", type=int, default=s.n_qubits)
    ap.add_argument("--q-layers", type=int, default=s.n_q_layers)
    ap.add_argument("--backend", default=s.quantum_backend)
    ap.add_argument("--no-flip", action="store_true")
    a = ap.parse_args()
    hp = {"n_qubits": a.qubits, "n_layers": a.q_layers, "backend": a.backend}
    cfg = TrainConfig(epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, seed=a.seed, device=a.device)
    out = a.out or s.model_path / "runs" / f"{a.kind}_{a.mode}_q{a.qubits}l{a.q_layers}_seed{a.seed}"
    if a.mode == "fine_tuning":
        run_train_finetune(a.kind, a.data, out, cfg, hp, image_size=s.image_size)
    else:
        run_train_head(a.kind, a.cache, out, a.data, cfg, hp, use_flip=not a.no_flip)
