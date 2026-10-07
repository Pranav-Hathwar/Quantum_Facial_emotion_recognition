# Quantum pipeline (PennyLane)

* **Reduction** – `Linear(2048 → n_qubits)`, then `π·tanh(x)` so each value is a valid rotation angle.
* **Encoding** – angle encoding: `R_y(x_i)` on qubit *i* (Chapter 12: |ψ⟩ = ∏ R_y(f_i)|0⟩).
* **VQC** – `L` layers of trainable `RX, RY, RZ` on every qubit followed by a CNOT ring; weights shape `(L, n, 3)`.
  Default 4 qubits × 2 layers = 24 trainable circuit parameters.
* **Measurement** – Pauli-Z expectation value of every qubit → `n` real numbers in [−1, 1].
* **Fusion** – concat(classical 256-d, quantum n-d) → Dropout → Linear(7) → softmax.
* **Training** – differentiable through PennyLane's `default.qubit` simulator (`diff_method="backprop"`, torch interface),
  so ordinary backprop updates both circuit and classical parameters.

Verification: `tests/test_quantum.py` compares the circuit output to an independent numpy state-vector implementation
and checks that gradients flow to the circuit weights. The circuit is a **classical simulation**; simulation cost grows
as 2ⁿ, and nothing here implies quantum speed-up. Hardware backends could be selected via `QUANTUM_BACKEND`, but were
not used or tested.
