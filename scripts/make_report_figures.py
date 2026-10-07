"""Report-ready comparison figures built from the measured ml/models/comparison.json.

    python scripts/make_report_figures.py            # -> docs/figures/*.png

Reads only what compare.py already wrote; it never recomputes or invents numbers. Produces:
  - metrics_bar.png          accuracy + macro-F1 per model, with std error bars over seeds
  - timing_bar.png           training time and per-face inference time per model (log scale)
  - confusion_matrices.png   seed-summed, row-normalised confusion matrix per model
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.config import get_settings
from ml.labels import EMOTIONS

KINDS = ("classical", "control", "hybrid")
COLORS = {"classical": "#4f8cff", "control": "#9aa0a6", "hybrid": "#ff6ea8"}


def _load(models_dir: Path) -> dict:
    data = json.loads((models_dir / "comparison.json").read_text())
    if not data.get("table"):
        raise SystemExit("comparison.json has no results yet; run compare.py first.")
    return data


def metrics_bar(table: dict, out: Path) -> None:
    kinds = [k for k in KINDS if k in table]
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]
    x = np.arange(len(metrics))
    width = 0.8 / len(kinds)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, k in enumerate(kinds):
        means = [table[k][m]["mean"] for m, _ in metrics]
        stds = [table[k][m]["std"] for m, _ in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4, label=table[k]["label"], color=COLORS[k])
    ax.set_xticks(x + width * (len(kinds) - 1) / 2)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (mean ± std over seeds)")
    ax.set_title("Classical vs Hybrid — FER2013 test split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def timing_bar(table: dict, out: Path) -> None:
    kinds = [k for k in KINDS if k in table]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.5))
    labels = [table[k]["label"] for k in kinds]
    pos = np.arange(len(kinds))
    a1.bar(pos, [table[k]["train_time_s"]["mean"] for k in kinds],
           yerr=[table[k]["train_time_s"]["std"] for k in kinds], capsize=4, color=[COLORS[k] for k in kinds])
    a1.set_ylabel("seconds")
    a1.set_title("Training time (per run)")
    a2.bar(pos, [table[k]["total_ms_per_face"]["mean"] for k in kinds], color=[COLORS[k] for k in kinds])
    a2.set_ylabel("ms / face")
    a2.set_title("Total inference time")
    for a in (a1, a2):
        a.set_xticks(pos)
        a.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def confusion_matrices(table: dict, out: Path) -> None:
    kinds = [k for k in KINDS if k in table]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4.6))
    if len(kinds) == 1:
        axes = [axes]
    labels = [e[:3].title() for e in EMOTIONS]
    for ax, k in zip(axes, kinds):
        cm = np.array(table[k]["confusion_matrix_sum"], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{table[k]['label']}\nacc {table[k]['accuracy']['mean']:.3f}", fontsize=9)
        ax.set_xticks(range(len(EMOTIONS)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(EMOTIONS)), labels, fontsize=7)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
        for i in range(len(EMOTIONS)):
            for j in range(len(EMOTIONS)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="row-normalised share")
    fig.suptitle("Confusion matrices (seed-summed, row-normalised)", y=1.02)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    s = get_settings()
    data = _load(s.model_path)
    table = data["table"]
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_bar(table, out_dir / "metrics_bar.png")
    timing_bar(table, out_dir / "timing_bar.png")
    confusion_matrices(table, out_dir / "confusion_matrices.png")
    print(f"Wrote figures to {out_dir}/ : metrics_bar.png, timing_bar.png, confusion_matrices.png")
