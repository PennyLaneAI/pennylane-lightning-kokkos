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
r"""
Helper functions for serializing quantum tapes.
"""
from typing import List, Tuple

import numpy as np
from pennylane import (
    BasisState,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    Identity,
    StatePrep,
    Rot,
)
from pennylane.operation import Tensor
from pennylane.ops.op_math import Adjoint
from pennylane.tape import QuantumTape

# Remove after the next release of PL
# Add from pennylane import matrix
import pennylane as qml

try:
    from pennylane_lightning_kokkos.lightning_kokkos_qubit_ops import (
        LightningKokkos_C128,
        LightningKokkos_C64,
        NamedObsKokkos_C64,
        NamedObsKokkos_C128,
        TensorProdObsKokkos_C64,
        TensorProdObsKokkos_C128,
        HamiltonianKokkos_C64,
        HamiltonianKokkos_C128,
        SparseHamiltonianKokkos_C64,
        SparseHamiltonianKokkos_C128,
        HermitianObsKokkos_C64,
        HermitianObsKokkos_C128,
    )
except ImportError as e:
    print(e)

pauli_name_map = {
    "I": "Identity",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
}


def _serialize_named_ob(o, wires_map: dict, use_csingle: bool):
    """Serializes an observable (Named)

    Args:
        o (Observable): the input observable (Named)
        wire_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        named_obs (NamedObsKokkos_C64 or NamedObsKokkos_C128): A Named observable object compatible with the C++ backend
    """
    named_obs = NamedObsKokkos_C64 if use_csingle else NamedObsKokkos_C128
    wires = [wires_map[w] for w in o.wires]
    if o.name == "Identity":
        wires = wires[:1]
    return named_obs(o.name, wires)


def _serialize_tensor_ob(ob, wires_map: dict, use_csingle: bool):
    """Serialize an observable (Tensor)

    Args:
        o (Observable): the input observable (Tensor)
        wire_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        tensor_obs (TensorProdObsKokkos_C64 or TensorProdObsKokkos_C128): A Tensor observable object compatible with the C++ backend
    """
    assert isinstance(ob, Tensor)

    if use_csingle:
        tensor_obs = TensorProdObsKokkos_C64
    else:
        tensor_obs = TensorProdObsKokkos_C128
    return tensor_obs([_serialize_ob(o, wires_map, use_csingle) for o in ob.obs])


def _serialize_hamiltonian(ob, wires_map: dict, use_csingle: bool):
    """Serialize an observable (Hamiltonian)

    Args:
        o (Observable): the input observable (Hamiltonian)
        wire_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        hamiltonian_obs (HamiltonianKokkos_C64 or HamiltonianKokkos_C128): A Hamiltonian observable object compatible with the C++ backend
    """
    if use_csingle:
        rtype = np.float32
        hamiltonian_obs = HamiltonianKokkos_C64
    else:
        rtype = np.float64
        hamiltonian_obs = HamiltonianKokkos_C128

    coeffs = np.array(ob.coeffs).astype(rtype)
    terms = [_serialize_ob(t, wires_map, use_csingle) for t in ob.ops]
    return hamiltonian_obs(coeffs, terms)


def _serialize_sparsehamiltonian(ob, wires_map: dict, use_csingle: bool):
    """Serialize an observable (Sparse Hamiltonian)

    Args:
        o (Observable): the input observable (Sparse Hamiltonian)
        wire_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        sparsehamiltonian_obs (SparseHamiltonianKokkos_C64 or SparseHamiltonianKokkos_C128): A Sparse Hamiltonian observable object compatible with the C++ backend
    """
    if use_csingle:
        ctype = np.complex64
        rtype = np.int32
        sparsehamiltonian_obs = SparseHamiltonianKokkos_C64
    else:
        ctype = np.complex128
        rtype = np.int64
        sparsehamiltonian_obs = SparseHamiltonianKokkos_C128

    spm = ob.sparse_matrix()
    data = np.array(spm.data).astype(ctype)
    indices = np.array(spm.indices).astype(rtype)
    offsets = np.array(spm.indptr).astype(rtype)

    wires = []
    wires_list = ob.wires.tolist()
    wires.extend([wires_map[w] for w in wires_list])

    return sparsehamiltonian_obs(data, indices, offsets, wires)


def _serialize_hermitian(ob, wires_map: dict, use_csingle: bool):
    """Serialize an observable (Hermitian)

    Args:
        o (Observable): the input observable (Hermitian)
        wire_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        hermitian_obs (HermitianObsKokkos_C64 or HermitianObsKokkos_C128): A Hermitian observable object compatible with the C++ backend
    """
    if use_csingle:
        rtype = np.float32
        hermitian_obs = HermitianObsKokkos_C64
    else:
        rtype = np.float64
        hermitian_obs = HermitianObsKokkos_C128

    data = qml.matrix(ob).astype(rtype).ravel(order="C")
    return hermitian_obs(data, ob.wires.tolist())


def _serialize_pauli_word(ob, wires_map: dict, use_csingle: bool):
    """Serialize a :class:`pennylane.pauli.PauliWord` into a Named or Tensor observable."""
    if use_csingle:
        named_obs = NamedObsKokkos_C64
        tensor_obs = TensorProdObsKokkos_C64
    else:
        named_obs = NamedObsKokkos_C128
        tensor_obs = TensorProdObsKokkos_C128

    if len(ob) == 1:
        wire, pauli = list(ob.items())[0]
        return named_obs(pauli_name_map[pauli], [wires_map[wire]])

    return tensor_obs(
        [named_obs(pauli_name_map[pauli], [wires_map[wire]]) for wire, pauli in ob.items()]
    )


def _serialize_pauli_sentence(ob, wires_map: dict, use_csingle: bool):
    """Serialize a :class:`pennylane.pauli.PauliSentence` into a Hamiltonian."""
    if use_csingle:
        rtype = np.float32
        hamiltonian_obs = HamiltonianKokkos_C64
    else:
        rtype = np.float64
        hamiltonian_obs = HamiltonianKokkos_C128

    pwords, coeffs = zip(*ob.items())
    terms = [_serialize_pauli_word(pw, wires_map, use_csingle) for pw in pwords]
    coeffs = np.array(coeffs).astype(rtype)
    return hamiltonian_obs(coeffs, terms)


def _serialize_ob(ob, wires_map, use_csingle):
    """Serialize an observable.
    Args:
        ob (Observable): the input observable
        wires_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128
    Returns:
        ObservableKokkos_C64 or ObservableKokkos_C128: An observable object compatible with the C++ backend
    """
    if isinstance(ob, Tensor):
        return _serialize_tensor_ob(ob, wires_map, use_csingle)
    elif ob.name == "Hamiltonian":
        return _serialize_hamiltonian(ob, wires_map, use_csingle)
    elif ob.name == "SparseHamiltonian":
        return _serialize_sparsehamiltonian(ob, wires_map, use_csingle)
    elif isinstance(ob, (PauliX, PauliY, PauliZ, Identity, Hadamard)):
        return _serialize_named_ob(ob, wires_map, use_csingle)
    elif ob._pauli_rep is not None:
        return _serialize_pauli_sentence(ob._pauli_rep, wires_map, use_csingle)
    elif ob.name == "Hermitian":
        raise TypeError(
            "Hermitian observables are not currently supported for adjoint differentiation. Please use Pauli-words only."
        )
    else:
        raise TypeError(f"Unknown observable found: {ob}. Please use Pauli-words only.")


def _serialize_observables(tape: QuantumTape, wires_map: dict, use_csingle: bool = False) -> List:
    """Serialize the observables of an input tape.
    Args:
        tape (QuantumTape): the input quantum tape
        wires_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128
    Returns:
        list(ObservableKokkos_C64 or ObservableKokkos_C128): A list of observable objects compatible with the C++ backend
    """

    return [_serialize_ob(ob, wires_map, use_csingle) for ob in tape.observables]


def _serialize_ops(
    tape: QuantumTape, wires_map: dict, use_csingle: bool = False
) -> Tuple[List[List[str]], List[np.ndarray], List[List[int]], List[bool], List[np.ndarray]]:
    """Serializes the operations of an input tape.

    The state preparation operations are not included.

    Args:
        tape (QuantumTape): the input quantum tape
        wires_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        Tuple[list, list, list, list, list]: A serialization of the operations, containing a list
        of operation names, a list of operation parameters, a list of observable wires, a list of
        inverses, and a list of matrices for the operations that do not have a dedicated kernel.
    """
    names = []
    params = []
    wires = []
    inverses = []
    mats = []

    uses_stateprep = False

    sv_py = LightningKokkos_C64 if use_csingle else LightningKokkos_C128

    for o in tape.operations:
        if isinstance(o, (BasisState, StatePrep)):
            uses_stateprep = True
            continue
        elif isinstance(o, Rot):
            op_list = o.expand().operations
        else:
            op_list = [o]

        for single_op in op_list:
            is_inverse = isinstance(single_op, Adjoint)

            name = single_op.name if not is_inverse else single_op.name[:-4]
            names.append(name)

            if getattr(sv_py, name, None) is None:
                params.append([])
                mats.append(qml.matrix(single_op))

                if is_inverse:
                    is_inverse = False
            else:
                params.append(single_op.parameters)
                mats.append([])

            wires_list = single_op.wires.tolist()
            wires.append([wires_map[w] for w in wires_list])
            inverses.append(is_inverse)

    return (names, params, wires, inverses, mats), uses_stateprep
