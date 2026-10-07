"""Model-agnostic training loop shared by the classical, control and hybrid models.

Same loop, same optimizer, same schedule for every model -> a fair comparison (Chapter 12: Adam-style
classical optimiser updating all trainable parameters, L(theta) = 1/N sum l(y_i, y_hat_i), theta <- theta - eta * grad).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    patience: int = 6              # early stopping on validation macro-F1
    use_class_weights: bool = True # FER2013 Disgust is rare
    label_smoothing: float = 0.0
    seed: int = 42
    device: str = "cpu"
    amp: bool = False              # mixed precision (CUDA only) - speeds up backbone fine-tuning


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig,
        out_dir: Path, class_weights: list[float] | None = None, meta: dict | None = None,
        log=print) -> dict:
    """Train, keep the best-validation-macro-F1 weights in out_dir/best.pt, return the history dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    dev = cfg.device
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    weight = torch.tensor(class_weights, dtype=torch.float32, device=dev) if (cfg.use_class_weights and class_weights) else None
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    use_amp = bool(cfg.amp) and str(dev).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"config": asdict(cfg), "meta": meta or {}, "epochs": []}
    best_f1, best_epoch, bad, train_time = -1.0, -1, 0, 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        sched.step()
        epoch_time = time.perf_counter() - t0
        train_time += epoch_time

        probs, labels = predict_probs(model, val_loader, dev)
        preds = probs.argmax(1)
        val_acc = float((preds == labels).mean())
        val_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        rec = {"epoch": epoch, "train_loss": loss_sum / total, "train_acc": correct / total,
               "val_acc": val_acc, "val_macro_f1": val_f1, "epoch_time_s": epoch_time}
        history["epochs"].append(rec)
        log(f"epoch {epoch:02d}/{cfg.epochs}  loss {rec['train_loss']:.4f}  train_acc {rec['train_acc']:.3f}  "
            f"val_acc {val_acc:.3f}  val_macroF1 {val_f1:.3f}  ({epoch_time:.1f}s)")

        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            torch.save({"state_dict": model.state_dict(), "meta": meta or {}, "epoch": epoch}, out_dir / "best.pt")
            log(f"  --> [NEW BEST] Val Macro-F1: {best_f1:.4f} (Saved checkpoint to {out_dir / 'best.pt'})")
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"  [STOP] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch} with Macro-F1: {best_f1:.4f})")
                break

    history.update(best_epoch=best_epoch, best_val_macro_f1=best_f1, train_time_s=train_time,
                   epochs_run=len(history["epochs"]))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history
