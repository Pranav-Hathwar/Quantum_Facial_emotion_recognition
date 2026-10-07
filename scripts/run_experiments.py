"""The full experiment in one command (after extract_features.py):
    classical (A) + control (A') + hybrid (B), several seeds each -> evaluate -> comparison.md

    python run_experiments.py --seeds 42 43 44 --epochs 30 --device cuda
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import get_settings
from ml.evaluation.compare import build_comparison
from ml.evaluation.evaluate import evaluate_run
from ml.training.run import run_train_head
from ml.training.trainer import TrainConfig

if __name__ == "__main__":
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=s.learning_rate)
    ap.add_argument("--device", default=s.device)
    ap.add_argument("--qubits", type=int, default=s.n_qubits)
    ap.add_argument("--q-layers", type=int, default=s.n_q_layers)
    ap.add_argument("--kinds", nargs="+", default=["classical", "control", "hybrid"])
    ap.add_argument("--data", type=Path, default=s.dataset_path)
    ap.add_argument("--cache", type=Path, default=s.model_path / "features")
    a = ap.parse_args()
    runs = s.model_path / "runs"
    hp = {"n_qubits": a.qubits, "n_layers": a.q_layers, "backend": s.quantum_backend}
    for seed in a.seeds:
        for kind in a.kinds:
            out = runs / (f"{kind}_feature_extractor_seed{seed}" if kind == "classical"
                          else f"{kind}_feature_extractor_q{a.qubits}l{a.q_layers}_seed{seed}")
            cfg = TrainConfig(epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, seed=seed, device=a.device)
            run_train_head(kind, a.cache, out, a.data, cfg, hp if kind != "classical" else None)
            m = evaluate_run(out, a.cache, "test", "cpu")
            print(f"==> {out.name}: test acc {m['accuracy']:.4f}  macroF1 {m['macro']['f1']:.4f}")
    print(build_comparison(runs)["summary"])
