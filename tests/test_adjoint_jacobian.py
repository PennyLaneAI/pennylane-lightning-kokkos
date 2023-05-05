# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the ``adjoint_jacobian`` method of LightningKokkos.
"""
import math
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import QNode, qnode
from scipy.stats import unitary_group
import pennylane_lightning_kokkos as plk
from pennylane_lightning_kokkos._serialize import _serialize_ob
from pennylane_lightning_kokkos.lightning_kokkos_qubit_ops import (
    InitializationSettings,
    NamedObsKokkos_C64,
    NamedObsKokkos_C128,
    TensorProdObsKokkos_C64,
    TensorProdObsKokkos_C128,
    HamiltonianKokkos_C64,
    HamiltonianKokkos_C128,
    SparseHamiltonianKokkos_C64,
    SparseHamiltonianKokkos_C128,
)

from pennylane import (
    QuantumFunctionError,
)


I, X, Y, Z = (
    np.eye(2),
    qml.PauliX.compute_matrix(),
    qml.PauliY.compute_matrix(),
    qml.PauliZ.compute_matrix(),
)


def Rx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * X


def Ry(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * Y


def Rz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * Z


class TestAdjointJacobian:
    """Tests for the adjoint_jacobian method"""

    from pennylane_lightning_kokkos import LightningKokkos as lk
    from pennylane_lightning import LightningQubit as lq

    @pytest.fixture(params=[None, InitializationSettings().set_num_threads(2)])
    def dev_kokkos(self, request):
        if request.param is None:
            return qml.device("lightning.kokkos", wires=3)
        else:
            return qml.device("lightning.kokkos", wires=3, kokkos_args=request.param)

    @pytest.fixture
    def dev_default(self):
        return qml.device("default.qubit", wires=3)

    def test_not_expval(self, dev_kokkos):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Adjoint differentiation method does"):
            dev_kokkos.adjoint_jacobian(tape)

    @pytest.mark.skip(reason="Warning not currently raised in favour of NotImplementedError")
    def test_finite_shots_warns(self):
        """Tests warning raised when finite shots specified"""

        dev = qml.device("lightning.kokkos", wires=1, shots=1)

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.warns(
            UserWarning,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            dev.adjoint_jacobian(tape)

    @pytest.mark.skipif(not lk._CPP_BINARY_AVAILABLE, reason="LightningKokkos support required")
    def test_unsupported_op(self, dev_kokkos):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""

        with qml.tape.QuantumTape() as tape:
            qml.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The CRot operation is not supported using the",
        ):
            dev_kokkos.adjoint_jacobian(tape)

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    @pytest.mark.skipif(not lk._CPP_BINARY_AVAILABLE, reason="LightningKokkos unsupported")
    def test_proj_unsupported(self, dev_kokkos):
        """Test if a QuantumFunctionError is raised for a Projector observable"""
        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0, 1], wires=[0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev_kokkos.adjoint_jacobian(tape)

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0], wires=[0]) @ qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev_kokkos.adjoint_jacobian(tape)

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    @pytest.mark.skipif(not lk._CPP_BINARY_AVAILABLE, reason="LightningKokkos unsupported")
    def test_unsupported_hermitian_expectation(self, dev_kokkos):
        obs = np.array([[1, 0], [0, -1]], dtype=np.complex128, requires_grad=False)

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.1, wires=(0,))
            qml.expval(qml.Hermitian(obs, wires=(0,)))

        with pytest.raises(
            qml.QuantumFunctionError,
        ):
            dev_kokkos.adjoint_jacobian(tape)

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.1, wires=(0,))
            qml.expval(qml.Hermitian(obs, wires=(0,)) @ qml.PauliZ(wires=1))

        with pytest.raises(
            qml.QuantumFunctionError,
        ):
            dev_kokkos.adjoint_jacobian(tape)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    def test_pauli_rotation_gradient(self, G, theta, tol, dev_default, dev_kokkos):
        """Tests that the automatic gradients of Pauli rotations are correct."""

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        calculated_val = dev_kokkos.adjoint_jacobian(tape)
        expected_val = dev_default.adjoint_jacobian(tape)

        assert np.allclose(calculated_val, expected_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_Rot_gradient(self, theta, tol, dev_default, dev_kokkos):
        """Tests that the device gradient of an arbitrary Euler-angle-parameterized gate is
        correct."""
        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        calculated_val = dev_kokkos.adjoint_jacobian(tape)
        expected_val = dev_default.adjoint_jacobian(tape)

        assert np.allclose(calculated_val, expected_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    def test_ry_gradient(self, par, tol, dev_kokkos):
        """Test that the gradient of the RY gate matches the exact analytic formula."""

        with qml.tape.QuantumTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        # gradients
        exact = np.cos(par)
        gtapes, fn = qml.gradients.param_shift(tape)
        grad_PS = fn(qml.execute(gtapes, dev_kokkos, gradient_fn=None))
        grad_A = dev_kokkos.adjoint_jacobian(tape)

        # different methods must agree
        assert np.allclose(grad_PS, exact, atol=tol, rtol=0)
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)

    def test_rx_gradient(self, tol, dev_kokkos):
        """Test that the gradient of the RX gate matches the known formula."""
        a = 0.7418

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.expval(qml.PauliZ(0))

        # circuit jacobians
        dev_jacobian = dev_kokkos.adjoint_jacobian(tape)
        expected_jacobian = -np.sin(a)
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_multiple_rx_gradient(self, tol, dev_kokkos):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            for idx in range(3):
                qml.expval(qml.PauliZ(idx))

        # circuit jacobians
        grad_A_kokkos = dev_kokkos.adjoint_jacobian(tape)
        gtapes, fn = qml.gradients.param_shift(tape)
        grad_PS_kokkos = fn(qml.execute(gtapes, dev_kokkos, gradient_fn=None))

        expected_jacobian = -np.diag(np.sin(params))
        assert np.allclose(grad_PS_kokkos, grad_A_kokkos, atol=tol, rtol=0)
        assert np.allclose(grad_A_kokkos, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qml, name) for name in qml.ops._qubit__ops__]
    ops = {qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.Rot}

    @pytest.mark.parametrize("obs", [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Identity])
    @pytest.mark.parametrize(
        "op",
        [
            qml.RX(0.4, wires=0),
            qml.RY(0.6, wires=0),
            qml.RZ(0.8, wires=0),
            qml.CRX(1.0, wires=[0, 1]),
            qml.CRY(2.0, wires=[0, 1]),
            qml.CRZ(3.0, wires=[0, 1]),
            qml.Rot(0.2, -0.1, 0.2, wires=0),
        ],
    )
    def test_gradients(self, op, obs, tol, dev_default, dev_kokkos):
        """Tests that the gradients of circuits match between the param-shift and device
        methods."""

        # op.num_wires and op.num_params must be initialized a priori
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.adjoint(qml.RY(0.5, wires=1), lazy=False)
            qml.CNOT(wires=[0, 1])

            qml.expval(obs(wires=0))
            qml.expval(qml.PauliZ(wires=1))

        tape.trainable_params = set(range(1, 1 + op.num_params))

        grad_PS = (lambda t, fn: fn(qml.execute(t, dev_kokkos, None)))(
            *qml.gradients.param_shift(tape)
        )
        grad_D = dev_kokkos.adjoint_jacobian(tape)

        assert np.allclose(grad_D, grad_PS, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters(self, tol, dev_kokkos):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        grad_D = dev_kokkos.adjoint_jacobian(tape)
        gtapes, fn = qml.gradients.param_shift(tape)
        grad_PS = fn(qml.execute(gtapes, dev_kokkos, gradient_fn=None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_PS, atol=tol, rtol=0)

    def test_use_device_state(self, tol, dev_kokkos):
        """Tests that when using the device state, the correct answer is still returned."""

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dM1 = dev_kokkos.adjoint_jacobian(tape)

        qml.execute([tape], dev_kokkos, None)
        dM2 = dev_kokkos.adjoint_jacobian(tape, use_device_state=True)

        assert np.allclose(dM1, dM2, atol=tol, rtol=0)

    def test_provide_starting_state(self, tol, dev_kokkos):
        """Tests provides correct answer when provided starting state."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dM1 = dev_kokkos.adjoint_jacobian(tape)

        qml.execute([tape], dev_kokkos, None)
        state_vector = np.zeros(2**dev_kokkos.num_wires).astype(dev_kokkos.C_DTYPE)
        dev_kokkos.syncD2H(state_vector)
        dM2 = dev_kokkos.adjoint_jacobian(tape, starting_state=state_vector)

        assert np.allclose(dM1, dM2, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "old_obs",
        [
            qml.PauliX(0) @ qml.PauliZ(1),
            qml.Hamiltonian([1.1], [qml.PauliZ(0)]),
            qml.Hamiltonian([1.1, 2.2], [qml.PauliZ(0), qml.PauliZ(1)]),
            qml.Hamiltonian([1.1, 2.2], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliX(1)]),
        ],
    )
    def test_op_arithmetic_is_supported(self, old_obs, dev_kokkos):
        """Tests that an arithmetic obs with a PauliRep are supported for adjoint_jacobian."""
        ops = [qml.RX(1.1, 0), qml.RY(2.2, 0), qml.RX(0.66, 1), qml.RY(1.23, 1)]
        new_obs = qml.pauli.pauli_sentence(old_obs).operation()
        old_tape = qml.tape.QuantumScript(ops, [qml.expval(old_obs)])
        new_tape = qml.tape.QuantumScript(ops, [qml.expval(new_obs)])
        old_res = dev_kokkos.adjoint_jacobian(old_tape)
        new_res = dev_kokkos.adjoint_jacobian(new_tape)
        assert qml.math.allequal(old_res, new_res)


class TestAdjointJacobianQNode:
    """Test QNode integration with the adjoint_jacobian method"""

    @pytest.fixture
    def dev_kokkos(self):
        return qml.device("lightning.kokkos", wires=2)

    def test_finite_shots_warning(self):
        """Tests that a warning is raised when computing the adjoint diff on a device with finite shots"""

        param = qml.numpy.array(0.1)
        dev = qml.device("lightning.kokkos", wires=1, shots=1)
        with pytest.warns(
            UserWarning,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):

            @qml.qnode(dev, diff_method="adjoint")
            def circ(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            UserWarning,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            qml.grad(circ)(param)

    def test_qnode(self, mocker, tol, dev_kokkos):
        """Test that specifying diff_method allows the adjoint method to be selected"""
        args = np.array([0.54, 0.1, 0.5], requires_grad=True)

        def circuit(x, y, z):
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.Rot(x, y, z, wires=0)

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

        qnode1 = QNode(circuit, dev_kokkos, diff_method="adjoint")
        spy = mocker.spy(dev_kokkos, "adjoint_jacobian")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()

        qnode2 = QNode(circuit, dev_kokkos, diff_method="parameter-shift")
        grad_fn = qml.grad(qnode2)
        grad_PS = grad_fn(*args)

        assert np.allclose(grad_A, grad_PS, atol=tol, rtol=0)

    thetas = np.linspace(-2 * np.pi, 2 * np.pi, 8)

    @pytest.mark.parametrize("reused_p", thetas**3 / 19)
    @pytest.mark.parametrize("other_p", thetas**2 / 1)
    def test_fanout_multiple_params(self, reused_p, other_p, tol, mocker, dev_kokkos):
        """Tests that the correct gradient is computed for qnodes which
        use the same parameter in multiple gates."""

        def expZ(state):
            return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

        extra_param = np.array(0.31, requires_grad=False)

        @qnode(dev_kokkos, diff_method="adjoint")
        def cost(p1, p2):
            qml.RX(extra_param, wires=[0])
            qml.RY(p1, wires=[0])
            qml.RZ(p2, wires=[0])
            qml.RX(p1, wires=[0])
            return qml.expval(qml.PauliZ(0))

        zero_state = np.array([1.0, 0.0])
        cost(reused_p, other_p)

        spy = mocker.spy(dev_kokkos, "adjoint_jacobian")

        # analytic gradient
        grad_fn = qml.grad(cost)
        grad_D = grad_fn(reused_p, other_p)

        spy.assert_called_once()

        # manual gradient
        grad_true0 = (
            expZ(
                Rx(reused_p) @ Rz(other_p) @ Ry(reused_p + np.pi / 2) @ Rx(extra_param) @ zero_state
            )
            - expZ(
                Rx(reused_p) @ Rz(other_p) @ Ry(reused_p - np.pi / 2) @ Rx(extra_param) @ zero_state
            )
        ) / 2
        grad_true1 = (
            expZ(
                Rx(reused_p + np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state
            )
            - expZ(
                Rx(reused_p - np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state
            )
        ) / 2
        expected = grad_true0 + grad_true1  # product rule

        assert np.allclose(grad_D[0], expected, atol=tol, rtol=0)

    def test_gradient_repeated_gate_parameters(self, mocker, tol, dev_kokkos):
        """Tests that repeated use of a free parameter in a multi-parameter gate yields correct
        gradients."""
        params = np.array([0.8, 1.3], requires_grad=True)

        def circuit(params):
            qml.RX(np.array(np.pi / 4, requires_grad=False), wires=[0])
            qml.Rot(params[1], params[0], 2 * params[0], wires=[0])
            return qml.expval(qml.PauliX(0))

        spy_analytic = mocker.spy(dev_kokkos, "adjoint_jacobian")

        cost = QNode(circuit, dev_kokkos, diff_method="parameter-shift")

        grad_fn = qml.grad(cost)
        grad_PS = grad_fn(params)

        spy_analytic.assert_not_called()

        cost = QNode(circuit, dev_kokkos, diff_method="adjoint")
        grad_fn = qml.grad(cost)
        grad_D = grad_fn(params)

        spy_analytic.assert_called_once()

        # the different methods agree
        assert np.allclose(grad_D, grad_PS, atol=tol, rtol=0)

    def test_interface_tf(self, dev_kokkos):
        """Test if gradients agree between the adjoint and parameter-shift methods when using the
        TensorFlow interface"""
        tf = pytest.importorskip("tensorflow")

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * tf.sqrt(params2), wires=[0])
            qml.RY(tf.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = tf.Variable(0.3, dtype=tf.float64)
        params2 = tf.Variable(0.4, dtype=tf.float64)

        qnode1 = QNode(f, dev_kokkos, interface="tf", diff_method="adjoint")
        qnode2 = QNode(f, dev_kokkos, interface="tf", diff_method="parameter-shift")

        with tf.GradientTape() as tape:
            res1 = qnode1(params1, params2)

        g1 = tape.gradient(res1, [params1, params2])

        with tf.GradientTape() as tape:
            res2 = qnode2(params1, params2)

        g2 = tape.gradient(res2, [params1, params2])

        assert np.allclose(g1, g2)

    def test_interface_torch(self, dev_kokkos):
        """Test if gradients agree between the adjoint and parameter-shift methods when using the
        Torch interface"""
        torch = pytest.importorskip("torch")

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * torch.sqrt(params2), wires=[0])
            qml.RY(torch.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = torch.tensor(0.3, requires_grad=True)
        params2 = torch.tensor(0.4, requires_grad=True)

        qnode1 = QNode(f, dev_kokkos, interface="torch", diff_method="adjoint")
        qnode2 = QNode(f, dev_kokkos, interface="torch", diff_method="parameter-shift")

        res1 = qnode1(params1, params2)
        res1.backward()

        grad_adjoint = params1.grad, params2.grad

        res2 = qnode2(params1, params2)
        res2.backward()

        grad_ps = params1.grad, params2.grad

        assert np.allclose(grad_adjoint, grad_ps, atol=1e-7)

    def test_interface_jax(self, dev_kokkos):
        """Test if the gradients agree between adjoint and parameter-shift methods in the
        jax interface"""
        jax = pytest.importorskip("jax")

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * jax.numpy.sqrt(params2), wires=[0])
            qml.RY(jax.numpy.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = jax.numpy.array(0.3)
        params2 = jax.numpy.array(0.4)

        qnode_adjoint = QNode(f, dev_kokkos, interface="jax", diff_method="adjoint")
        qnode_ps = QNode(f, dev_kokkos, interface="jax", diff_method="parameter-shift")

        grad_adjoint = jax.grad(qnode_adjoint)(params1, params2)
        grad_ps = jax.grad(qnode_ps)(params1, params2)

        assert np.allclose(grad_adjoint, grad_ps, atol=1e-7)


def test_qchem_expvalcost_correct():
    """EvpvalCost with qchem Hamiltonian work corectly"""
    from pennylane import qchem

    symbols = ["Li", "H"]
    geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527])
    H, qubits = qchem.molecular_hamiltonian(
        symbols, geometry, active_electrons=2, active_orbitals=5
    )
    active_electrons = 2
    hf_state = qchem.hf_state(active_electrons, qubits)

    def circuit_1(params, wires):
        qml.BasisState(hf_state, wires=wires)
        qml.RX(params[0], wires=0)
        qml.RY(params[0], wires=1)
        qml.RZ(params[0], wires=2)
        qml.Hadamard(wires=1)

    diff_method = "adjoint"
    dev_lig = qml.device("lightning.kokkos", wires=qubits)
    cost_fn_lig = qml.ExpvalCost(circuit_1, H, dev_lig, optimize=False, diff_method=diff_method)
    circuit_gradient_lig = qml.grad(cost_fn_lig, argnum=0)
    params = np.array([0.123], requires_grad=True)
    grads_lig = circuit_gradient_lig(params)

    dev_def = qml.device("default.qubit", wires=qubits)
    cost_fn_def = qml.ExpvalCost(circuit_1, H, dev_def, optimize=False, diff_method=diff_method)
    circuit_gradient_def = qml.grad(cost_fn_def, argnum=0)
    params = np.array([0.123], requires_grad=True)
    grads_def = circuit_gradient_def(params)

    assert np.allclose(grads_lig, grads_def)


def circuit_ansatz(params, wires):
    """Circuit ansatz containing all the parametrized gates"""
    qml.QubitStateVector(unitary_group.rvs(2**4, random_state=0)[0], wires=wires)
    qml.RX(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.adjoint(qml.RX(params[2], wires=wires[2]))
    qml.RZ(params[0], wires=wires[3])
    qml.CRX(params[3], wires=[wires[3], wires[0]])
    qml.PhaseShift(params[4], wires=wires[2])
    qml.CRY(params[5], wires=[wires[2], wires[1]])
    qml.adjoint(qml.CRZ(params[5], wires=[wires[0], wires[3]]))
    qml.adjoint(qml.PhaseShift(params[6], wires=wires[0]))
    qml.Rot(params[6], params[7], params[8], wires=wires[0])
    qml.MultiRZ(params[11], wires=[wires[0], wires[1]])
    qml.CPhase(params[12], wires=[wires[3], wires[2]])
    qml.IsingXX(params[13], wires=[wires[1], wires[0]])
    qml.IsingYY(params[14], wires=[wires[3], wires[2]])
    qml.IsingZZ(params[14], wires=[wires[2], wires[1]])
    qml.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qml.DoubleExcitation(params[25], wires=[wires[2], wires[0], wires[1], wires[3]])
    qml.SingleExcitationPlus(params[26], wires=[wires[0], wires[2]])
    qml.SingleExcitationMinus(params[27], wires=[wires[0], wires[2]])
    qml.DoubleExcitationMinus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])
    qml.RX(params[28], wires=wires[0])
    qml.RX(params[29], wires=wires[1])
    qml.adjoint(qml.Rot(params[8], params[8], params[9], wires=wires[1]))
    qml.DoubleExcitationPlus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])


@pytest.mark.parametrize(
    "returns",
    [
        qml.PauliZ(0),
        qml.PauliX(2),
        qml.PauliZ(0) @ qml.PauliY(3),
        qml.Hadamard(2),
        qml.Hadamard(3) @ qml.PauliZ(2),
        qml.PauliX(0) @ qml.PauliY(3),
        qml.PauliY(0) @ qml.PauliY(2) @ qml.PauliY(3),
    ],
)
def test_integration(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""
    dev_lightning = qml.device("lightning.qubit", wires=range(4))
    dev_kokkos = qml.device("lightning.kokkos", wires=range(4))

    def circuit(params):
        circuit_ansatz(params, wires=range(4))
        return np.array([qml.expval(returns), qml.expval(qml.PauliY(1))])

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_kokkos = qml.QNode(circuit, dev_kokkos, diff_method="adjoint")
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="adjoint")

    j_kokkos = qml.jacobian(qnode_kokkos)(params)
    j_lightning = qml.jacobian(qnode_lightning)(params)

    assert np.allclose(j_kokkos, j_lightning, atol=1e-5)


custom_wires = ["alice", 3.14, -1, 0]


@pytest.mark.parametrize(
    "returns",
    [
        qml.PauliZ(custom_wires[0]),
        qml.PauliX(custom_wires[2]),
        qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
        qml.Hadamard(custom_wires[2]),
        qml.Hadamard(custom_wires[3]) @ qml.PauliZ(custom_wires[2]),
        # qml.Projector([0, 1], wires=[custom_wires[0], custom_wires[2]]) @ qml.Hadamard(custom_wires[3])
        # qml.Projector([0, 0], wires=[custom_wires[2], custom_wires[0]])
        qml.PauliX(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
        qml.PauliY(custom_wires[0]) @ qml.PauliY(custom_wires[2]) @ qml.PauliY(custom_wires[3]),
        # qml.Hermitian(np.array([[0,1],[1,0]], requires_grad=False), wires=custom_wires[0]),
        # qml.Hermitian(np.kron(qml.PauliY.matrix, qml.PauliZ.matrix), wires=[custom_wires[3], custom_wires[2]]),
        # qml.Hermitian(np.array([[0,1],[1,0]], requires_grad=False), wires=custom_wires[0]) @ qml.PauliZ(custom_wires[2]),
    ],
)
def test_integration_custom_wires(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_lightning = qml.device("lightning.qubit", wires=custom_wires)
    dev_kokkos = qml.device("lightning.kokkos", wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return np.array([qml.expval(returns), qml.expval(qml.PauliY(custom_wires[1]))])

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_kokkos = qml.QNode(circuit, dev_kokkos, diff_method="adjoint")
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="adjoint")

    j_kokkos = qml.jacobian(qnode_kokkos)(params)
    j_lightning = qml.jacobian(qnode_lightning)(params)

    assert np.allclose(j_kokkos, j_lightning, atol=1e-7)


@pytest.mark.parametrize(
    "returns",
    [
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0]),
        qml.Hermitian(
            np.kron(qml.PauliY.compute_matrix(), qml.PauliZ.compute_matrix()),
            wires=[custom_wires[3], custom_wires[2]],
        ),
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0])
        @ qml.PauliZ(custom_wires[2]),
    ],
)
def test_fail_adjoint_Hermitian(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_kokkos = qml.device("lightning.kokkos", wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns)

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_kokkos = qml.QNode(circuit, dev_kokkos, diff_method="adjoint")

    with pytest.raises(
        qml._device.DeviceError,
        match="Observable Hermitian not supported on device",
    ):
        j_kokkos = qml.jacobian(qnode_kokkos)(params)


@pytest.mark.parametrize(
    "returns",
    [
        0.6
        * qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0])
        @ qml.PauliX(wires=custom_wires[1]),
    ],
)
def test_fail_adjoint_mixed_Hamiltonian_Hermitian(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_kokkos = qml.device("lightning.kokkos", wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns)

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_kokkos = qml.QNode(circuit, dev_kokkos, diff_method="adjoint")

    with pytest.raises(Exception):
        j_kokkos = qml.jacobian(qnode_kokkos)(params)


def test_adjoint_Hamiltonian():
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    coeffs = [0.5, -0.25, 1.0]
    obs = [
        qml.PauliX(custom_wires[0]) @ qml.PauliX(custom_wires[1]),
        qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[2]),
        qml.PauliZ(custom_wires[3]),
    ]
    hamiltonian = qml.Hamiltonian(coeffs, obs)

    dev_kokkos = qml.device("lightning.kokkos", wires=custom_wires)
    dev_default = qml.device("default.qubit", wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(hamiltonian)

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_kokkos = qml.QNode(circuit, dev_kokkos, diff_method="adjoint")
    qnode_default = qml.QNode(circuit, dev_default, diff_method="parameter-shift")

    j_kokkos = qml.jacobian(qnode_kokkos)(params)
    j_default = qml.jacobian(qnode_default)(params)

    assert np.allclose(j_kokkos, j_default)


@pytest.mark.parametrize(
    "returns",
    [
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliX(wires=custom_wires[0]) @ qml.PauliZ(wires=custom_wires[1])],
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [2.0],
                [qml.PauliX(wires=custom_wires[2]) @ qml.PauliZ(wires=custom_wires[0])],
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [1.1],
                [qml.PauliX(wires=custom_wires[0]) @ qml.PauliZ(wires=custom_wires[2])],
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
    ],
)
def test_adjoint_SparseHamiltonian(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_kokkos = qml.device("lightning.kokkos", wires=custom_wires)
    dev_default = qml.device("default.qubit", wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns)

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_kokkos = qml.QNode(circuit, dev_kokkos, diff_method="adjoint")
    qnode_default = qml.QNode(circuit, dev_default, diff_method="parameter-shift")

    j_kokkos = qml.jacobian(qnode_kokkos)(params)
    j_default = qml.jacobian(qnode_default)(params)

    assert np.allclose(j_kokkos, j_default)


@pytest.mark.parametrize(
    "returns",
    [
        (qml.PauliZ(custom_wires[0]),),
        (qml.PauliZ(custom_wires[0]), qml.PauliZ(custom_wires[1])),
        (qml.PauliZ(custom_wires[0]), qml.PauliZ(custom_wires[1]), qml.PauliZ(custom_wires[3])),
        (
            qml.PauliZ(custom_wires[0]),
            qml.PauliZ(custom_wires[1]),
            qml.PauliZ(custom_wires[3]),
            qml.PauliZ(custom_wires[2]),
        ),
        (
            qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
            qml.PauliZ(custom_wires[1]) @ qml.PauliY(custom_wires[2]),
        ),
        (qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]), qml.PauliZ(custom_wires[1])),
    ],
)
def test_integration_custom_wires_batching(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_lightning = qml.device("lightning.qubit", wires=custom_wires)
    dev_kokkos = qml.device("lightning.kokkos", wires=custom_wires, batch_obs=True)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return [qml.expval(r) for r in returns] + [qml.expval(qml.PauliY(custom_wires[1]))]

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_kokkos = qml.QNode(circuit, dev_kokkos, diff_method="adjoint")
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="adjoint")

    def convert_to_array_kokkos(params):
        return np.hstack(qnode_kokkos(params))

    def convert_to_array_lightning(params):
        return np.hstack(qnode_lightning(params))

    j_kokkos = qml.jacobian(convert_to_array_kokkos)(params)
    j_lightning = qml.jacobian(convert_to_array_lightning)(params)

    assert np.allclose(j_kokkos, j_lightning, atol=1e-7)


@pytest.mark.parametrize(
    "obs,obs_type_c64,obs_type_c128",
    [
        (qml.PauliZ(0), NamedObsKokkos_C64, NamedObsKokkos_C128),
        (qml.PauliZ(0) @ qml.PauliZ(1), TensorProdObsKokkos_C64, TensorProdObsKokkos_C128),
        (qml.Hadamard(0), NamedObsKokkos_C64, NamedObsKokkos_C128),
        (qml.Hamiltonian([1], [qml.PauliZ(0)]), HamiltonianKokkos_C64, HamiltonianKokkos_C128),
        (
            qml.PauliZ(0) @ qml.Hadamard(1) @ (0.1 * (qml.PauliZ(2) + qml.PauliX(3))),
            HamiltonianKokkos_C64,
            HamiltonianKokkos_C128,
        ),
        (
            qml.SparseHamiltonian(qml.Hamiltonian([1], [qml.PauliZ(0)]).sparse_matrix(), wires=[0]),
            SparseHamiltonianKokkos_C64,
            SparseHamiltonianKokkos_C128,
        ),
    ],
)
@pytest.mark.parametrize("use_csingle", [True, False])
def test_obs_returns_expected_type(obs, obs_type_c64, obs_type_c128, use_csingle):
    """Tests that observables get serialized to the expected type."""
    obs_type = obs_type_c64 if use_csingle else obs_type_c128
    assert isinstance(_serialize_ob(obs, dict(enumerate(obs.wires)), use_csingle), obs_type)


@pytest.mark.parametrize(
    "bad_obs",
    [
        qml.Hermitian(np.eye(2), wires=0),
        qml.sum(qml.PauliZ(0), qml.Hadamard(1)),
        qml.Projector([0], wires=0),
        qml.PauliZ(0) @ qml.Projector([0], wires=1),
        qml.sum(qml.Hadamard(0), qml.PauliX(1)),
    ],
)
@pytest.mark.parametrize("use_csingle", [True, False])
def test_obs_not_supported_for_adjoint_diff(bad_obs, use_csingle):
    """Tests observables that can't be serialized for adjoint-differentiation."""
    with pytest.raises(TypeError, match="Please use Pauli-words only."):
        _serialize_ob(bad_obs, dict(enumerate(bad_obs.wires)), use_csingle)
