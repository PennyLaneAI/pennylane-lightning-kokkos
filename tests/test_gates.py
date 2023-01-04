# Copyright 2021 Xanadu Quantum Technologies Inc.

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
import itertools

import numpy as np
import pennylane as qml
import pytest

from pennylane_lightning_kokkos import LightningKokkos


@pytest.fixture
def op(op_name):
    ops_list = {
        "RX": qml.RX(0.123, wires=0),
        "RY": qml.RY(1.434, wires=0),
        "RZ": qml.RZ(2.774, wires=0),
        "S": qml.S(wires=0),
        # "SX": qml.SX(wires=0),
        "T": qml.T(wires=0),
        "CNOT": qml.CNOT(wires=[0, 1]),
        "CZ": qml.CZ(wires=[0, 1]),
        "CY": qml.CY(wires=[0, 1]),
        "SWAP": qml.SWAP(wires=[0, 1]),
        # "ISWAP": qml.ISWAP(wires=[0, 1]),
        # "SISWAP": qml.SISWAP(wires=[0, 1]),
        # "SQISW": qml.SQISW(wires=[0, 1]),
        "CSWAP": qml.CSWAP(wires=[0, 1, 2]),
        "PauliRot": qml.PauliRot(0.123, "Y", wires=0),
        "IsingXX": qml.IsingXX(0.123, wires=[0, 1]),
        "IsingXY": qml.IsingXY(0.123, wires=[0, 1]),
        "IsingYY": qml.IsingYY(0.123, wires=[0, 1]),
        "IsingZZ": qml.IsingZZ(0.123, wires=[0, 1]),
        # "Identity": qml.Identity(wires=0),
        # "Rot": qml.Rot(0.123, 0.456, 0.789, wires=0),
        "Toffoli": qml.Toffoli(wires=[0, 1, 2]),
        "PhaseShift": qml.PhaseShift(2.133, wires=0),
        "ControlledPhaseShift": qml.ControlledPhaseShift(1.777, wires=[0, 2]),
        # "CPhase": qml.CPhase(1.777, wires=[0, 2]),
        "MultiRZ": qml.MultiRZ(0.112, wires=[1, 2, 3]),
        "CRX": qml.CRX(0.836, wires=[2, 3]),
        "CRY": qml.CRY(0.721, wires=[2, 3]),
        "CRZ": qml.CRZ(0.554, wires=[2, 3]),
        "Hadamard": qml.Hadamard(wires=0),
        "PauliX": qml.PauliX(wires=0),
        "PauliY": qml.PauliY(wires=0),
        "PauliZ": qml.PauliZ(wires=0),
        "CRot": qml.CRot(0.123, 0.456, 0.789, wires=[0, 1]),
        "DiagonalQubitUnitary": qml.DiagonalQubitUnitary(np.array([1.0, 1.0j]), wires=1),
        "ControlledQubitUnitary": qml.ControlledQubitUnitary(
            np.eye(2) * 1j, wires=[0], control_wires=[2]
        ),
        # "MultiControlledX": qml.MultiControlledX(wires=(0, 1, 2), control_values="01"),
        "SingleExcitation": qml.SingleExcitation(0.123, wires=[0, 3]),
        "SingleExcitationPlus": qml.SingleExcitationPlus(0.123, wires=[0, 3]),
        "SingleExcitationMinus": qml.SingleExcitationMinus(0.123, wires=[0, 3]),
        "DoubleExcitation": qml.DoubleExcitation(0.123, wires=[0, 1, 2, 3]),
        "DoubleExcitationPlus": qml.DoubleExcitationPlus(0.123, wires=[0, 1, 2, 3]),
        "DoubleExcitationMinus": qml.DoubleExcitationMinus(0.123, wires=[0, 1, 2, 3]),
        # "QFT": qml.QFT(wires=0),
        # "QubitSum": qml.QubitSum(wires=[0, 1, 2]),
        # "QubitCarry": qml.QubitCarry(wires=[0, 1, 2, 3]),
        # "QubitUnitary": qml.QubitUnitary(np.eye(2) * 1j, wires=0),
    }
    return ops_list.get(op_name)


@pytest.mark.parametrize("op_name", LightningKokkos.operations)
def test_gate_unitary_correct(op, op_name):
    """Test if lightning.kokkos correctly applies gates by reconstructing the unitary matrix and
    comparing to the expected version"""

    if op_name in ("BasisState", "QubitStateVector"):
        pytest.skip("Skipping operation because it is a state preparation")
    if op_name in (
        "ControlledQubitUnitary",
        "QubitUnitary",
        "MultiControlledX",
        "DiagonalQubitUnitary",
    ):
        pytest.skip("Skipping operation.")  # These are tested in the device test-suite
    if op == None:
        pytest.skip("Skipping operation.")

    wires = op.num_wires

    if wires == -1:  # This occurs for operations that do not have a predefined number of wires
        wires = 4

    dev = qml.device("lightning.kokkos", wires=wires)
    num_params = op.num_params
    p = [0.1] * num_params

    op = getattr(qml, op_name)

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        op(*p, wires=range(wires))
        return qml.state()

    unitary = np.zeros((2**wires, 2**wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(np.array(input))
        unitary[:, i] = out

    unitary_expected = qml.matrix(op(*p, wires=range(wires)))

    assert np.allclose(unitary, unitary_expected)


@pytest.mark.parametrize("op_name", LightningKokkos.operations)
def test_inverse_unitary_correct(op, op_name):
    """Test if lightning.kokkos correctly applies inverse gates by reconstructing the unitary matrix
    and comparing to the expected version"""

    if op_name in ("BasisState", "QubitStateVector"):
        pytest.skip("Skipping operation because it is a state preparation")
    if op_name in (
        "ControlledQubitUnitary",
        "QubitUnitary",
        "MultiControlledX",
        "DiagonalQubitUnitary",
    ):
        pytest.skip("Skipping operation.")  # These are tested in the device test-suite
    if op == None:
        pytest.skip("Skipping operation.")

    wires = op.num_wires

    if wires == -1:  # This occurs for operations that do not have a predefined number of wires
        wires = 4

    dev = qml.device("lightning.kokkos", wires=wires)
    num_params = op.num_params
    p = [0.1] * num_params

    print(op_name, p)

    op = getattr(qml, op_name)

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        op(*p, wires=range(wires)).inv()
        return qml.state()

    unitary = np.zeros((2**wires, 2**wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(np.array(input))
        unitary[:, i] = out

    unitary_expected = qml.matrix(op(*p, wires=range(wires)).inv())

    assert np.allclose(unitary, unitary_expected)
