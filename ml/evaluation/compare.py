"""Aggregate all evaluated runs into the classical-vs-hybrid comparison (mean +/- std over seeds + McNemar test).

Writes <runs_dir>/../comparison.json and comparison.md. The summary sentence is generated mechanically from the
numbers and never claims an advantage that the statistics do not support.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from ml.labels import EMOTIONS

LABELS = {"classical": "Classical ResNet50 (A)", "control": "ResNet50 + classical layer (A')",
          "hybrid": "Hybrid ResNet50 + Quantum VQC (B)"}


def _ms(values: list[float]) -> dict:
    v = np.array([x for x in values if x is not None], dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))} if len(v) else {"mean": None, "std": None, "n": 0}


def mcnemar(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray) -> dict:
    """Exact McNemar test on paired test predictions (same test images)."""
    a_ok, b_ok = preds_a == labels, preds_b == labels
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = only_a + only_b
    p = float(binomtest(only_a, n, 0.5).pvalue) if n > 0 else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": p}


def build_comparison(runs_dir: Path, include_smoke: bool = False) -> dict:
    """Runs whose backbone used random (non-pretrained) weights are smoke tests and are excluded by default."""
    runs_dir = Path(runs_dir)
    groups: dict[str, list[dict]] = defaultdict(list)
    dirs: dict[str, list[Path]] = defaultdict(list)
    for m in sorted(runs_dir.glob("*/metrics.json")):
        data = json.loads(m.read_text())
        if data.get("backbone", {}).get("pretrained") is False and not include_smoke:
            continue
        if data.get("split") == "test":
            groups[data["model_kind"]].append(data)
            dirs[data["model_kind"]].append(m.parent)

    table = {}
    for kind, runs in groups.items():
        table[kind] = {
            "label": LABELS.get(kind, kind), "n_runs": len(runs), "seeds": [r.get("seed") for r in runs],
            "accuracy": _ms([r["accuracy"] for r in runs]),
            "macro_precision": _ms([r["macro"]["precision"] for r in runs]),
            "macro_recall": _ms([r["macro"]["recall"] for r in runs]),
            "macro_f1": _ms([r["macro"]["f1"] for r in runs]),
            "weighted_f1": _ms([r["weighted"]["f1"] for r in runs]),
            "train_time_s": _ms([r["training"]["train_time_s"] for r in runs]),
            "head_ms_per_face": _ms([r["inference"]["head_ms_per_face"] for r in runs]),
            "total_ms_per_face": _ms([r["inference"]["total_ms_per_face"] for r in runs]),
            "head_parameters": runs[0]["parameters"]["head_total"],
            "quantum_parameters": runs[0]["parameters"]["quantum_circuit"],
            "per_class_f1": {e: _ms([r["per_class"][e]["f1"] for r in runs]) for e in EMOTIONS},
            "per_class": {e: {m: _ms([r["per_class"][e][m] for r in runs]) for m in ("precision", "recall", "f1", "accuracy")}
                          for e in EMOTIONS},
            "support": {e: runs[0]["per_class"][e]["support"] for e in EMOTIONS},
            "confusion_matrix_sum": np.sum([np.array(r["confusion_matrix"]) for r in runs], axis=0).tolist(),
        }

    tests = {}
    if "classical" in dirs and "hybrid" in dirs:
        # pair runs by seed
        def preds(d: Path):
            z = np.load(d / "test_predictions.npz")
            return z["probs"].argmax(1), z["labels"]
        seeds_a = {json.loads((d / "metrics.json").read_text())["seed"]: d for d in dirs["classical"]}
        seeds_b = {json.loads((d / "metrics.json").read_text())["seed"]: d for d in dirs["hybrid"]}
        for seed in sorted(set(seeds_a) & set(seeds_b)):
            pa, y = preds(seeds_a[seed]); pb, _ = preds(seeds_b[seed])
            tests[f"seed_{seed}"] = mcnemar(pa, pb, y)

    result = {"table": table, "mcnemar_classical_vs_hybrid": tests, "summary": _summary(table, tests)}
    out = runs_dir.parent
    (out / "comparison.json").write_text(json.dumps(result, indent=2))
    (out / "comparison.md").write_text(_markdown(result))
    return result


def _summary(table: dict, tests: dict) -> str:
    if "classical" not in table or "hybrid" not in table:
        return "Both the classical and the hybrid model must be trained and evaluated before they can be compared."
    a, b = table["classical"]["accuracy"], table["hybrid"]["accuracy"]
    diff = b["mean"] - a["mean"]
    n = min(a["n"], b["n"])
    pvals = [t["p_value"] for t in tests.values()]
    significant = bool(pvals) and all(p < 0.05 for p in pvals)
    spread = max(a["std"], b["std"])
    if abs(diff) <= spread or not significant:
        verdict = "no consistent, statistically supported accuracy difference between the two models"
    elif diff > 0:
        verdict = "the hybrid model scored higher on this test set (paired McNemar p<0.05 for every seed)"
    else:
        verdict = "the classical baseline scored higher on this test set (paired McNemar p<0.05 for every seed)"
    return (f"Over {n} seed(s) on the FER2013 test split: classical accuracy {a['mean']:.4f}, hybrid {b['mean']:.4f} "
            f"(difference {diff:+.4f}); {verdict}. Results are from classical simulation of the quantum circuit on one dataset "
            "and do not demonstrate quantum advantage or speed-up.")


def _fmt(v: dict, pct: bool = False, digits: int = 4) -> str:
    if v["mean"] is None:
        return "-"
    s = f"{v['mean']:.{digits}f}" + (f" ± {v['std']:.{digits}f}" if v["n"] > 1 else "")
    return s


def _markdown(res: dict) -> str:
    kinds = [k for k in ("classical", "control", "hybrid") if k in res["table"]]
    lines = ["# Classical vs Hybrid Quantum — measured results (FER2013 test split)", "",
             res["summary"], "", "| Metric | " + " | ".join(res["table"][k]["label"] for k in kinds) + " |",
             "|---|" + "---|" * len(kinds)]
    rows = [("Accuracy", "accuracy"), ("Macro precision", "macro_precision"), ("Macro recall", "macro_recall"),
            ("Macro F1", "macro_f1"), ("Weighted F1", "weighted_f1"), ("Training time (s)", "train_time_s"),
            ("Head inference (ms/face)", "head_ms_per_face"), ("Total inference (ms/face)", "total_ms_per_face")]
    for name, key in rows:
        lines.append(f"| {name} | " + " | ".join(_fmt(res["table"][k][key], digits=2 if "time" in key or "ms" in key else 4) for k in kinds) + " |")
    lines.append("| Head parameters | " + " | ".join(f"{res['table'][k]['head_parameters']:,}" for k in kinds) + " |")
    lines.append("| of which quantum circuit | " + " | ".join(f"{res['table'][k]['quantum_parameters']:,}" for k in kinds) + " |")
    lines += ["", "Values are mean ± std over seeds. Inference time of the hybrid includes classical simulation of the circuit."]
    return "\n".join(lines) + "\n"
