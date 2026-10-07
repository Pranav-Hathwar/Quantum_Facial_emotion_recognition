"""Build / save / load the three comparable heads from a small hyper-parameter dict."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ml.classical.head import ClassicalHead
from ml.quantum.quantum_model import ControlHead, HybridHead

KINDS = ("classical", "control", "hybrid")


def build_head(kind: str, hp: dict | None = None) -> nn.Module:
    hp = dict(hp or {})
    hp.setdefault("in_features", 2048)
    hp.setdefault("hidden", 256)
    hp.setdefault("dropout", 0.3)
    if kind == "classical":
        return ClassicalHead(hp["in_features"], hp["hidden"], hp["dropout"])
    if kind == "control":
        return ControlHead(hp["in_features"], hp["hidden"], hp["dropout"], hp.get("n_qubits", 4))
    if kind == "hybrid":
        return HybridHead(hp["in_features"], hp["hidden"], hp["dropout"], hp.get("n_qubits", 4),
                          hp.get("n_layers", 2), hp.get("backend", "default.qubit"))
    raise ValueError(f"kind must be one of {KINDS}")


def load_head(ckpt_path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = build_head(meta["kind"], meta.get("hyperparameters"))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta


def load_serving_model(ckpt_path: Path, device: str = "cpu", pretrained_backbone: bool = True,
                       weights: str = "v1") -> tuple[nn.Module, nn.Module, dict]:
    """Return (backbone, head, meta) ready for inference from either training mode.

    feature_extractor runs: fresh ImageNet-pretrained frozen backbone + trained head.
    fine_tuning runs:       backbone and head both restored from the checkpoint.
    """
    from ml.classical.backbone import ResNet50Backbone

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    head = build_head(meta["kind"], meta.get("hyperparameters"))
    if meta.get("mode") == "fine_tuning":
        backbone = ResNet50Backbone(pretrained=False, mode="feature_extractor")
        backbone.net.load_state_dict({k[len("backbone.net."):]: v for k, v in ckpt["state_dict"].items()
                                      if k.startswith("backbone.net.")})
        head.load_state_dict({k[len("head."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("head.")})
    else:
        backbone = ResNet50Backbone(pretrained=pretrained_backbone, mode="feature_extractor", weights=weights)
        head.load_state_dict(ckpt["state_dict"])
    return backbone.eval().to(device), head.eval().to(device), meta
