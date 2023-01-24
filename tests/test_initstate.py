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
Unit tests for the :mod:`pennylane_lightning_kokkos.LightningKokkos` device.
"""
# pylint: disable=protected-access,cell-var-from-loop
import math

import numpy as np
import pennylane as qml
import pytest
from pennylane.wires import Wires
import pennylane_lightning_kokkos

INV_SQRT2 = 1 / np.sqrt(2)

# functions for creating different states used in testing
def basis_state(index, nr_wires):
    rho = np.zeros((2**nr_wires), dtype=np.complex128)
    rho[index] = 1
    return rho


class TestApplyBasisState:
    """Unit tests for the method `_apply_basis_state"""

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_all_ones(self, nr_wires, tol):
        """Tests that the state |11...1> is applied correctly"""
        dev = qml.device("lightning.kokkos", wires=nr_wires)
        state = np.ones(nr_wires)
        dev._apply_basis_state_kokkos(state, wires=Wires(range(nr_wires)))
        b_state = basis_state(2**nr_wires - 1, nr_wires)

        state_vector = np.zeros(2**dev.num_wires).astype(dev.C_DTYPE)
        dev.syncD2H(state_vector)

        assert np.allclose(state_vector, b_state, atol=tol, rtol=0)

    fixed_states = [[3, np.array([0, 1, 1])], [5, np.array([1, 0, 1])], [6, np.array([1, 1, 0])]]

    @pytest.mark.parametrize("state", fixed_states)
    def test_fixed_states(self, state, tol):
        """Tests that different basis states are applied correctly"""
        nr_wires = 3
        dev = qml.device("lightning.kokkos", wires=nr_wires)
        dev._apply_basis_state_kokkos(state[1], wires=Wires(range(nr_wires)))
        b_state = basis_state(state[0], nr_wires)

        state_vector = np.zeros(2**dev.num_wires).astype(dev.C_DTYPE)
        dev.syncD2H(state_vector)

        assert np.allclose(state_vector, b_state, atol=tol, rtol=0)

    def test_wrong_dim(self):
        """Checks that an error is raised if state has the wrong dimension"""
        dev = qml.device("lightning.kokkos", wires=3)
        state = np.ones(2)
        with pytest.raises(ValueError, match="BasisState parameter and wires"):
            dev._apply_basis_state_kokkos(state, wires=Wires(range(3)))

    def test_not_01(self):
        """Checks that an error is raised if state doesn't have entries in {0,1}"""
        dev = qml.device("lightning.kokkos", wires=2)
        state = np.array([INV_SQRT2, INV_SQRT2])
        with pytest.raises(ValueError, match="BasisState parameter must"):
            dev._apply_basis_state_kokkos(state, wires=Wires(range(2)))


class TestApplyStateVector:
    """Unit tests for the method `_apply_state_vector_kokkos()`"""

    def test_full_subsystem(self):
        """Test applying a state vector to the full subsystem"""
        dev = qml.device("lightning.kokkos", wires=["a", "b", "c"])
        state = np.array([1, 0, 0, 0, 1, 0, 1, 1]) / 2.0
        state_wires = qml.wires.Wires(["a", "b", "c"])

        dev._apply_state_vector_kokkos(state=state, device_wires=state_wires)

        state_vector = np.zeros(2**dev.num_wires).astype(dev.C_DTYPE)
        dev.syncD2H(state_vector)

        assert np.all(state_vector == state)

    def test_partial_subsystem(self):
        """Test applying a state vector to a subset of wires of the full subsystem"""

        dev = qml.device("lightning.kokkos", wires=["a", "b", "c"])
        state = np.array([1, 0, 1, 0]) / np.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        dev._apply_state_vector_kokkos(state=state, device_wires=state_wires)

        state_vector = np.zeros(2**dev.num_wires).astype(dev.C_DTYPE)
        dev.syncD2H(state_vector)

        res = np.sum(np.reshape(state_vector, [2] * dev.num_wires), axis=(1,)).flatten()

        assert np.all(res == state)
