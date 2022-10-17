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
Unit tests for the generate_samples method of the :mod:`pennylane_lightning_gpu.LightningGPU` device.
"""
import pytest

import numpy as np
import pennylane as qml
import pennylane_lightning_kokkos as plk

try:
    from pennylane_lightning_kokkos import LightningKokkos

    if not LightningKokkos._CPP_BINARY_AVAILABLE:
        raise ImportError("PennyLane-Lightning-Kokkos is unsupported on this platform")
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "PennyLane-Lightning-Kokkos is unsupported on this platform. Skipping.",
        allow_module_level=True,
    )
np.random.seed(42)


class TestSample:
    """Tests that samples are properly calculated."""

    def test_sample_dimensions(self):
        """Tests if the samples returned by sample have
        the correct dimensions
        """
        dev = qml.device("lightning.kokkos", wires=2, shots=1000, c_dtype=np.complex128)

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        dev.apply([qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])])

        dev.shots = 10
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()
        s1 = dev.sample(qml.PauliZ(wires=[0]))
        assert np.array_equal(s1.shape, (10,))

        dev.reset()
        dev.shots = 12
        dev._wires_measured = {1}
        dev._samples = dev.generate_samples()
        s2 = dev.sample(qml.PauliZ(wires=[1]))
        assert np.array_equal(s2.shape, (12,))

        dev.reset()
        dev.shots = 17
        dev._wires_measured = {0, 1}
        dev._samples = dev.generate_samples()
        s3 = dev.sample(qml.PauliX(0) @ qml.PauliZ(1))
        assert np.array_equal(s3.shape, (17,))

    def test_sample_values(self, tol):
        """Tests if the samples returned by sample have
        the correct values
        """

        dev = qml.device("lightning.kokkos", wires=2, shots=1000, c_dtype=np.complex128)

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset

        # dev._state = dev._asarray(dev._state)

        kokkos_ctor = plk.lightning_kokkos._kokkos_dtype(dev.C_DTYPE)
        dev._kokkos_state = kokkos_ctor(np.array(dev._state).astype(dev.C_DTYPE))

        dev.apply([qml.RX(1.5708, wires=[0])])
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()

        s1 = dev.sample(qml.PauliZ(0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)
