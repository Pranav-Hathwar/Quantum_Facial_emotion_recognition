import numpy as np
import pytest
import torch

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from ml.quantum.circuit import circuit_spec, draw_circuit
from ml.quantum.encoding import scale_to_angles
from ml.quantum.quantum_model import ControlHead, HybridHead, QuantumLayer


# ---- an independent pure-numpy statevector simulator of the same circuit -------------------------------
def _apply_1q(state, gate, wire, n):
    state = state.reshape([2] * n)
    state = np.moveaxis(np.tensordot(gate, state, axes=([1], [wire])), 0, wire)
    return state.reshape(-1)


def _cnot(state, c, t, n):
    s = state.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[c] = 1
    sub = s[tuple(idx)]
    t_axis = t if t < c else t - 1
    s[tuple(idx)] = np.flip(sub, axis=t_axis)
    return s.reshape(-1)


def _rx(a): return np.array([[np.cos(a / 2), -1j * np.sin(a / 2)], [-1j * np.sin(a / 2), np.cos(a / 2)]])
def _ry(a): return np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]], dtype=complex)
def _rz(a): return np.array([[np.exp(-1j * a / 2), 0], [0, np.exp(1j * a / 2)]])


def reference_expvals(angles, weights, n):
    state = np.zeros(2 ** n, dtype=complex); state[0] = 1
    for i in range(n):
        state = _apply_1q(state, _ry(angles[i]), i, n)
    for layer in weights:
        for i in range(n):
            for gate, a in zip((_rx, _ry, _rz), layer[i]):
                state = _apply_1q(state, gate(a), i, n)
        for i in range(n):
            state = _cnot(state, i, (i + 1) % n, n)
    probs = np.abs(state.reshape([2] * n)) ** 2
    out = []
    for w in range(n):
        p = np.moveaxis(probs, w, 0).reshape(2, -1).sum(1)
        out.append(p[0] - p[1])
    return np.array(out)


def test_vqc_matches_independent_statevector_simulation():
    torch.manual_seed(0)
    layer = QuantumLayer(4, 2)
    angles = torch.rand(5, 4) * 2 * np.pi - np.pi
    z = layer(angles).detach().numpy()
    for b in range(5):
        ref = reference_expvals(angles[b].numpy().astype(float), layer.weights.detach().numpy().astype(float), 4)
        assert np.allclose(z[b], ref, atol=1e-5), (z[b], ref)


def test_output_is_expectation_values_in_range_and_batched():
    layer = QuantumLayer(4, 2)
    z = layer(torch.randn(8, 4))
    assert z.shape == (8, 4) and bool((z.abs() <= 1 + 1e-6).all())
    single = layer(torch.randn(8, 4)[:1])
    assert single.shape == (1, 4)


def test_gradients_reach_quantum_weights_and_inputs():
    layer = QuantumLayer(4, 2)
    x = torch.randn(3, 4, requires_grad=True)
    layer(x).sum().backward()
    assert layer.weights.grad is not None and layer.weights.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_angle_scaling_bounds():
    a = scale_to_angles(torch.tensor([-1e6, 0.0, 1e6]))
    assert float(a.abs().max()) <= np.pi + 1e-6


def test_circuit_drawing_and_spec():
    text = draw_circuit(4, 2)
    assert "RY" in text and "RX" in text and "RZ" in text and "╭●" in text or "●" in text
    spec = circuit_spec(4, 2)
    assert spec["n_trainable_parameters"] == 24 and spec["stages"][-1]["stage"] == "measurement"


def test_hybrid_head_shapes_and_param_split():
    h = HybridHead(in_features=64, hidden=16, n_qubits=4, n_layers=2)
    f = torch.randn(6, 64)
    parts = h.forward_with_intermediates(f)
    assert parts["logits"].shape == (6, 7) and parts["quantum"].shape == (6, 4) and parts["angles"].shape == (6, 4)
    assert h.quantum.weights.numel() == 24
    c = ControlHead(in_features=64, hidden=16, n_qubits=4)
    assert c(f).shape == (6, 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_hybrid_head_trains_on_cuda():
    # regression: default.qubit allocates its state on the CPU, which used to crash with cuda inputs
    model = HybridHead(in_features=16, hidden=8, n_qubits=3, n_layers=1).cuda()
    logits = model(torch.randn(4, 16, device="cuda"))
    assert logits.device.type == "cuda" and logits.shape == (4, 7)
    logits.sum().backward()
    grad = model.quantum.weights.grad
    assert grad is not None and grad.device.type == "cuda" and float(grad.abs().sum()) > 0
