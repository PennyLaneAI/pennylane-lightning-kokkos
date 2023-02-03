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
Unit tests for the expval method of the :mod:`pennylane_lightning_kokkos.LightningKokkos` device.
"""
import math

import numpy as np
import pennylane as qml
import pytest
from pennylane_lightning_kokkos import LightningKokkos


class TestSparseHamiltonianExpval:
    """Tests for the expval function"""

    def test_sparse_hamiltionan_expval(self, qubit_device_3_wires, tol):
        dev = LightningKokkos(wires=3, c_dtype=np.complex128)
        obs = qml.Identity(0) @ qml.PauliX(1) @ qml.PauliY(2)

        obs1 = qml.Identity(1)

        H = qml.Hamiltonian([1.0, 1.0], [obs1, obs])

        state_vector = np.array(
            [
                0.0 + 0.0j,
                0.0 + 0.1j,
                0.1 + 0.1j,
                0.1 + 0.2j,
                0.2 + 0.2j,
                0.3 + 0.3j,
                0.3 + 0.4j,
                0.4 + 0.5j,
            ],
            dtype=np.complex64,
        )

        dev.syncH2D(state_vector)
        Hmat = qml.utils.sparse_hamiltonian(H)
        H_sparse = qml.SparseHamiltonian(Hmat, wires=range(3))

        res = dev.expval(H_sparse)
        expected = 1

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0) @ qml.Identity(1), 0.00000000000000000],
            [qml.Identity(0) @ qml.PauliX(1), -0.19866933079506122],
            [qml.PauliY(0) @ qml.Identity(1), -0.38941834230865050],
            [qml.Identity(0) @ qml.PauliY(1), 0.00000000000000000],
            [qml.PauliZ(0) @ qml.Identity(1), 0.92106099400288520],
            [qml.Identity(0) @ qml.PauliZ(1), 0.98006657784124170],
        ],
    )
    def test_sparse_Pauli_words(self, cases, tol):
        """Test expval of some simple sparse Hamiltonian"""
        dev = LightningKokkos(wires=2, c_dtype=np.complex128)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(
                qml.SparseHamiltonian(
                    qml.utils.sparse_hamiltonian(qml.Hamiltonian([1], [cases[0]])), wires=[0, 1]
                )
            )

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0) @ qml.Identity(1), 0.00000000000000000],
            [qml.Identity(0) @ qml.PauliX(1), -0.19866933079506122],
            [qml.PauliY(0) @ qml.Identity(1), -0.38941834230865050],
            [qml.Identity(0) @ qml.PauliY(1), 0.00000000000000000],
            [qml.Hermitian([[1, 0], [0, -1]], wires=0) @ qml.Identity(1), 0.92106099400288520],
            [qml.Identity(0) @ qml.PauliZ(1), 0.98006657784124170],
            [
                qml.Hermitian([[1, 0], [0, 1]], wires=0)
                @ qml.Hermitian([[1, 0], [0, -1]], wires=1),
                0.98006657784124170,
            ],
        ],
    )
    def test_sparse_arbitrary(self, cases, tol):
        """Test expval of some simple sparse Hamiltonian"""
        dev = LightningKokkos(wires=2, c_dtype=np.complex128)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(
                qml.SparseHamiltonian(
                    qml.utils.sparse_hamiltonian(qml.Hamiltonian([1], [cases[0]])), wires=[0, 1]
                )
            )

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)
