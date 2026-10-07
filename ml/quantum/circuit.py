"""The full variational quantum circuit as a PennyLane QNode, plus drawing / description helpers.

|psi(theta)> = U_VQC(theta) U_enc(F) |0>^n        output Z = (<Z_0>, ..., <Z_{n-1}>) in [-1, 1]^n
Runs on a classical simulator (default.qubit) - this is a real quantum-circuit simulation, not a neural network.
"""
from __future__ import annotations

import pennylane as qml
import torch

from ml.quantum.encoding import angle_encoding
from ml.quantum.layers import variational_block
from ml.quantum.measurements import pauli_z_expectations


def make_device(n_qubits: int, backend: str = "default.qubit"):
    return qml.device(backend, wires=n_qubits)


def diff_method_for(backend: str) -> str:
    # exact backprop through the simulator is only available on the pure-Python/torch simulators
    return "backprop" if backend in {"default.qubit", "default.mixed"} else "parameter-shift"


def build_qnode(n_qubits: int, n_layers: int, backend: str = "default.qubit"):
    dev = make_device(n_qubits, backend)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="torch", diff_method=diff_method_for(backend))
    def circuit(inputs, weights):
        angle_encoding(inputs, wires)          # |psi>      = U_enc(F)|0>
        variational_block(weights, wires)      # |psi(th)>  = U_VQC(th)|psi>
        return pauli_z_expectations(wires)     # z_k        = <psi(th)|Z_k|psi(th)>

    return circuit


def draw_circuit(n_qubits: int, n_layers: int, backend: str = "default.qubit") -> str:
    """ASCII drawing of the circuit (for the Quantum Circuit page / docs)."""
    qnode = build_qnode(n_qubits, n_layers, backend)
    inputs = torch.zeros(n_qubits)
    weights = torch.zeros(n_layers, n_qubits, 3)
    return qml.draw(qnode, decimals=None, show_all_wires=True)(inputs, weights)


def circuit_spec(n_qubits: int, n_layers: int) -> dict:
    """Machine-readable description (the React page draws the circuit from this)."""
    columns: list[dict] = [{"stage": "encoding", "gates": [{"gate": "RY", "wire": i, "param": f"f{i}"} for i in range(n_qubits)]}]
    for layer in range(n_layers):
        cols = [{"gate": g, "wire": i, "param": f"θ[{layer},{i},{k}]"}
                for k, g in enumerate(("RX", "RY", "RZ")) for i in range(n_qubits)]
        cnots = [{"gate": "CNOT", "control": i, "target": (i + 1) % n_qubits} for i in range(n_qubits)] if n_qubits > 1 else []
        columns.append({"stage": f"variational_layer_{layer + 1}", "gates": cols, "entanglement": cnots})
    columns.append({"stage": "measurement", "gates": [{"gate": "expval(PauliZ)", "wire": i} for i in range(n_qubits)]})
    return {
        "n_qubits": n_qubits, "n_layers": n_layers,
        "n_trainable_parameters": n_layers * n_qubits * 3,
        "encoding": "Angle encoding (R_y per feature)",
        "variational": "RX-RY-RZ rotations + CNOT ring per layer",
        "measurement": "Pauli-Z expectation value per qubit",
        "stages": columns,
    }
