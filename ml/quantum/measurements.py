"""Quantum measurement (Chapter 12 step d.6): z_k = <psi(theta)| O_k |psi(theta)> with O_k = Pauli-Z on qubit k."""
from __future__ import annotations

from typing import Sequence

import pennylane as qml


def pauli_z_expectations(wires: Sequence[int]):
    return [qml.expval(qml.PauliZ(w)) for w in wires]
