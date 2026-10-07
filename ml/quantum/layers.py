"""Trainable variational layers (Chapter 12 step d.3): rotation gates + entanglement layers.

One layer =  RX(a) RY(b) RZ(c) on every qubit  ->  ring of CNOTs (0->1, 1->2, ..., n-1->0).
Trainable weights have shape (n_layers, n_qubits, 3).
"""
from __future__ import annotations

from typing import Sequence

import pennylane as qml


def entangling_ring(wires: Sequence[int]) -> None:
    n = len(wires)
    if n < 2:
        return
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def variational_layer(layer_weights, wires: Sequence[int]) -> None:
    for i, w in enumerate(wires):
        qml.RX(layer_weights[i, 0], wires=w)
        qml.RY(layer_weights[i, 1], wires=w)
        qml.RZ(layer_weights[i, 2], wires=w)
    entangling_ring(wires)


def variational_block(weights, wires: Sequence[int]) -> None:
    """U_VQC(theta): stack of `weights.shape[0]` variational layers."""
    for layer in range(weights.shape[0]):
        variational_layer(weights[layer], wires)
