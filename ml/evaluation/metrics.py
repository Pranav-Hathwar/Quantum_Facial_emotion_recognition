"""Classification metrics for the 7 emotion classes. Everything shown in the dashboard comes from here."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ml.labels import EMOTIONS, NUM_CLASSES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)[:3]
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)[:3]
    row_sums = cm.sum(axis=1)
    per_class_acc = np.divide(cm.diagonal(), row_sums, out=np.zeros(NUM_CLASSES), where=row_sums > 0)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
        "per_class": {
            EMOTIONS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "accuracy": float(per_class_acc[i]), "support": int(s[i])}
            for i in labels
        },
        "confusion_matrix": cm.tolist(),
        "class_order": list(EMOTIONS),
    }
