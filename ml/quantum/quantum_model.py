"""Torch modules wrapping the VQC (the quantum part) and the hybrid / control heads built around it.

Hybrid (Model B), Chapter 12 steps c-e:
    f (2048-d ResNet50)  --Linear-->  n features  --pi*tanh-->  angles  --VQC-->  z (n expectation values)
    f                    --Linear+ReLU-->  c (256 classical features)
    concat(c, z) -> Dropout -> Linear(7)     (softmax at inference)

Control (Model A'): identical, but the VQC is replaced by tanh(Linear(n, n)) - a classical layer of similar
size - so differences between B and A' can be attributed to the quantum circuit and not to extra parameters.
"""
from __future__ import annotations

import time

import torch
from torch import nn

from ml.labels import NUM_CLASSES
from ml.quantum.circuit import build_qnode, circuit_spec, draw_circuit
from ml.quantum.encoding import scale_to_angles


class QuantumLayer(nn.Module):
    """angles (B, n) -> expectation values (B, n). Trainable: weights (n_layers, n_qubits, 3)."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, backend: str = "default.qubit"):
        super().__init__()
        self.n_qubits, self.n_layers, self.backend = n_qubits, n_layers, backend
        self.weights = nn.Parameter(0.3 * torch.randn(n_layers, n_qubits, 3))
        self.qnode = build_qnode(n_qubits, n_layers, backend)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        # The PennyLane simulator allocates its state on the CPU, so the circuit always runs there; the .to()
        # transfers are differentiable, so gradients still reach parameters living on the GPU.
        x = angles.to("cpu", torch.float64) if self.backend == "default.qubit" else angles.cpu()
        out = self.qnode(x, self.weights.cpu())
        z = torch.stack(list(out), dim=-1) if isinstance(out, (tuple, list)) else out
        return z.to(angles.device, angles.dtype)

    def spec(self) -> dict:
        return circuit_spec(self.n_qubits, self.n_layers)

    def draw(self) -> str:
        return draw_circuit(self.n_qubits, self.n_layers, self.backend)


class HybridHead(nn.Module):
    kind = "hybrid"

    def __init__(self, in_features: int = 2048, hidden: int = 256, dropout: float = 0.3, n_qubits: int = 4,
                 n_layers: int = 2, backend: str = "default.qubit", num_classes: int = NUM_CLASSES):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU())      # classical features
        self.reduce = nn.Linear(in_features, n_qubits)                            # dimensionality reduction
        self.quantum = QuantumLayer(n_qubits, n_layers, backend)                  # encoding + VQC + measurement
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden + n_qubits, num_classes)                      # feature fusion + classifier

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(features)["logits"]

    def forward_with_intermediates(self, features: torch.Tensor, timings: dict | None = None) -> dict:
        """Every stage of the hybrid pipeline. If `timings` is given, per-stage milliseconds are written into it."""
        t0 = time.perf_counter()
        classical = self.proj(features)
        reduced = self.reduce(features)
        angles = scale_to_angles(reduced)
        t1 = time.perf_counter()
        quantum = self.quantum(angles)              # encoding + VQC + measurement are ONE circuit execution
        t2 = time.perf_counter()
        fused = torch.cat([classical, quantum], dim=1)
        logits = self.out(self.dropout(fused))
        t3 = time.perf_counter()
        if timings is not None:
            timings.update(feature_reduction_ms=(t1 - t0) * 1000, quantum_circuit_ms=(t2 - t1) * 1000,
                           fusion_classification_ms=(t3 - t2) * 1000)
        return {"classical": classical, "reduced": reduced, "angles": angles, "quantum": quantum, "logits": logits}


class ControlHead(nn.Module):
    """Model A': same wiring as HybridHead but with a classical tanh layer instead of the VQC."""
    kind = "control"

    def __init__(self, in_features: int = 2048, hidden: int = 256, dropout: float = 0.3, n_qubits: int = 4,
                 num_classes: int = NUM_CLASSES, **_ignored):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU())
        self.reduce = nn.Linear(in_features, n_qubits)
        self.mix = nn.Linear(n_qubits, n_qubits)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden + n_qubits, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.mix(scale_to_angles(self.reduce(features))))
        return self.out(self.dropout(torch.cat([self.proj(features), z], dim=1)))
