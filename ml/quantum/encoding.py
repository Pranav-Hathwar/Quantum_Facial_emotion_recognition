"""Quantum feature encoding (Chapter 12 step d.2).

Angle encoding:  |psi> = U_enc(F)|0>^(x)d = prod_i R_y(f_i)|0>
One classical feature per qubit. Features are first squashed to (-pi, pi) with pi*tanh(x) so the rotation
angles stay numerically reasonable no matter how large the reduced ResNet features are.
"""
from __future__ import annotations

import math
from typing import Sequence

import pennylane as qml
import torch


def scale_to_angles(x: torch.Tensor) -> torch.Tensor:
    """Map arbitrary real features to rotation angles in (-pi, pi)."""
    return math.pi * torch.tanh(x)


def angle_encoding(features, wires: Sequence[int]) -> None:
    """Apply R_y(f_i) to qubit i. `features` has shape (n,) or (batch, n); batching is handled by PennyLane."""
    for i, w in enumerate(wires):
        qml.RY(features[..., i], wires=w)
