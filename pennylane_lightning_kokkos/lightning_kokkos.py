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
This module contains the :class:`~.LightningKokkos` class, a PennyLane simulator device that
interfaces with Kokkos-enabled calculations to run efficiently on different kinds of shared memory
hardware systems, such as AMD and Nvidia GPUs, or many-core CPUs. 
"""
from typing import List
from warnings import warn
from itertools import product

import numpy as np
from pennylane import (
    active_return,
    math,
    QubitDevice,
    BasisState,
    QubitStateVector,
    DeviceError,
    Projector,
    Hermitian,
    Rot,
    QuantumFunctionError,
    QubitStateVector,
)
from pennylane_lightning import LightningQubit
from pennylane.operation import Tensor, Operation
from pennylane.measurements import Expectation, MeasurementProcess, State
from pennylane.ops.op_math import Adjoint

from pennylane.wires import Wires

# tolerance for numerical errors
tolerance = 1e-10

import pennylane as qml
from ._version import __version__

try:
    from .lightning_kokkos_qubit_ops import (
        InitializationSettings,
        LightningKokkos_C128,
        LightningKokkos_C64,
        AdjointJacobianKokkos_C128,
        AdjointJacobianKokkos_C64,
        print_configuration,
    )

    from ._serialize import _serialize_observables, _serialize_ops

    CPP_BINARY_AVAILABLE = True
except (ImportError, ValueError, PLException) as e:
    warn(str(e), UserWarning)
    CPP_BINARY_AVAILABLE = False


def _kokkos_dtype(dtype):
    if dtype not in [np.complex128, np.complex64]:
        raise ValueError(f"Data type is not supported for state-vector computation: {dtype}")
    return LightningKokkos_C128 if dtype == np.complex128 else LightningKokkos_C64


def _kokkos_configuration():
    return print_configuration()


allowed_operations = {
    "Identity",
    "BasisState",
    "QubitStateVector",
    "QubitUnitary",
    "ControlledQubitUnitary",
    "MultiControlledX",
    "DiagonalQubitUnitary",
    "PauliX",
    "PauliY",
    "PauliZ",
    "MultiRZ",
    "Hadamard",
    "S",
    "Adjoint(S)",
    "T",
    "Adjoint(T)",
    "SX",
    "Adjoint(SX)",
    "CNOT",
    "SWAP",
    "ISWAP",
    "PSWAP",
    "Adjoint(ISWAP)",
    "SISWAP",
    "Adjoint(SISWAP)",
    "SQISW",
    "CSWAP",
    "Toffoli",
    "CY",
    "CZ",
    "PhaseShift",
    "ControlledPhaseShift",
    "CPhase",
    "RX",
    "RY",
    "RZ",
    "Rot",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "IsingXX",
    "IsingYY",
    "IsingZZ",
    "IsingXY",
    "SingleExcitation",
    "SingleExcitationPlus",
    "SingleExcitationMinus",
    "DoubleExcitation",
    "DoubleExcitationPlus",
    "DoubleExcitationMinus",
    "QubitCarry",
    "QubitSum",
    "OrbitalRotation",
    "QFT",
    "ECR",
}


if CPP_BINARY_AVAILABLE:

    class LightningKokkos(QubitDevice):
        """PennyLane-Lightning-Kokkos device.
        Args:
            wires (int): the number of wires to initialize the device with
            sync (bool): immediately sync with host-sv after applying operations
            c_dtype: Datatypes for statevector representation. Must be one of ``np.complex64`` or ``np.complex128``.
            kokkos_args (InitializationSettings): binding for Kokkos::InitializationSettings (threading parameters).
        """

        name = "PennyLane plugin for Kokkos-backed Lightning device"
        short_name = "lightning.kokkos"
        pennylane_requires = ">=0.30"
        version = __version__
        author = "Xanadu Inc."
        _CPP_BINARY_AVAILABLE = True
        kokkos_config = {}

        operations = allowed_operations
        observables = {
            "PauliX",
            "PauliY",
            "PauliZ",
            "Hadamard",
            "SparseHamiltonian",
            "Hamiltonian",
            "Identity",
            "Sum",
            "Prod",
            "SProd",
        }

        def __init__(
            self,
            wires,
            *,
            sync=True,
            c_dtype=np.complex128,
            shots=None,
            batch_obs=False,
            kokkos_args=None,
        ):
            if c_dtype is np.complex64:
                r_dtype = np.float32
                self.use_csingle = True
            elif c_dtype is np.complex128:
                r_dtype = np.float64
                self.use_csingle = False
            else:
                raise TypeError(f"Unsupported complex Type: {c_dtype}")
            super().__init__(wires, shots=shots, r_dtype=r_dtype, c_dtype=c_dtype)
            if kokkos_args is None:
                self._kokkos_state = _kokkos_dtype(c_dtype)(self.num_wires)
            elif isinstance(kokkos_args, InitializationSettings):
                self._kokkos_state = _kokkos_dtype(c_dtype)(self.num_wires, kokkos_args)
            else:
                raise TypeError("Argument kokkos_args must be of type InitializationSettings.")
            self._sync = sync

            if not LightningKokkos.kokkos_config:
                LightningKokkos.kokkos_config = _kokkos_configuration()

        def reset(self):
            super().reset()
            # init the state vector to |00..0>
            self._kokkos_state.resetKokkos()  # Sync reset

        def syncH2D(self, state_vector):
            """Copy the state vector data on host provided by the user to the state vector on the device
            Args:
                state_vector(array[complex]): the state vector array on host.
            **Example**
            >>> dev = qml.device('lightning.kokkos', wires=3)
            >>> obs = qml.Identity(0) @ qml.PauliX(1) @ qml.PauliY(2)
            >>> obs1 = qml.Identity(1)
            >>> H = qml.Hamiltonian([1.0, 1.0], [obs1, obs])
            >>> state_vector = np.array([0.0 + 0.0j, 0.0 + 0.1j, 0.1 + 0.1j, 0.1 + 0.2j,
                0.2 + 0.2j, 0.3 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j,], dtype=np.complex64,)
            >>> dev.syncH2D(state_vector)
            >>> res = dev.expval(H)
            >>> print(res)
            1.0
            """
            self._kokkos_state.HostToDevice(state_vector.ravel(order="C"))

        def syncD2H(self, state_vector):
            """Copy the state vector data on device to a state vector on the host provided by the user
            Args:
                state_vector(array[complex]): the state vector array on host
            **Example**
            >>> dev = qml.device('lightning.kokkos', wires=1)
            >>> dev.apply([qml.PauliX(wires=[0])])
            >>> state_vector = np.zeros(2**dev.num_wires).astype(dev.C_DTYPE)
            >>> dev.syncD2H(state_vector)
            >>> print(state_vector)
            [0.+0.j 1.+0.j]
            """
            self._kokkos_state.DeviceToHost(state_vector.ravel(order="C"))

        @classmethod
        def capabilities(cls):
            capabilities = super().capabilities().copy()
            capabilities.update(
                model="qubit",
                supports_inverse_operations=True,
                supports_analytic_computation=True,
                supports_finite_shots=False,
                supports_broadcasting=False,
                returns_state=True,
            )
            capabilities.pop("passthru_devices", None)
            return capabilities

        @property
        def stopping_condition(self):
            """.BooleanFn: Returns the stopping condition for the device. The returned
            function accepts a queuable object (including a PennyLane operation
            and observable) and returns ``True`` if supported by the device."""

            def accepts_obj(obj):
                if obj.name == "QFT" and len(obj.wires) < 6:
                    return True
                if obj.name == "GroverOperator" and len(obj.wires) < 13:
                    return True
                return (not isinstance(obj, qml.tape.QuantumTape)) and getattr(
                    self, "supports_operation", lambda name: False
                )(obj.name)

            return qml.BooleanFn(accepts_obj)

        @property
        def state(self):
            """Copy the state vector data from the device to the host. A state vector Numpy array is explicitly allocated on the host to store and return the data.
            **Example**
            >>> dev = qml.device('lightning.kokkos', wires=1)
            >>> dev.apply([qml.PauliX(wires=[0])])
            >>> print(dev.state)
            [0.+0.j 1.+0.j]
            """
            state = np.zeros(2**self.num_wires, dtype=self.C_DTYPE)
            state = self._asarray(state, dtype=self.C_DTYPE)
            self.syncD2H(state)
            return state

        def _create_basis_state_kokkos(self, index):
            """Return a computational basis state over all wires.
            Args:
                index (int): integer representing the computational basis state
            Returns:
                array[complex]: complex array of shape ``[2]*self.num_wires``
                representing the statevector of the basis state
            Note: This function does not support broadcasted inputs yet.
            """
            self._kokkos_state.setBasisState(index)

        def _apply_state_vector_kokkos(self, state, device_wires):
            """Initialize the internal state vector in a specified state.
            Args:
                state (array[complex]): normalized input state of length ``2**len(wires)``
                device_wires (Wires): wires that get initialized in the state
            """

            # translate to wire labels used by device
            device_wires = self.map_wires(device_wires)
            dim = 2 ** len(device_wires)

            state = self._asarray(state, dtype=self.C_DTYPE)
            output_shape = [2] * self.num_wires

            if not qml.math.is_abstract(state):
                norm = qml.math.linalg.norm(state, axis=-1, ord=2)
                if not qml.math.allclose(norm, 1.0, atol=tolerance):
                    raise ValueError("Sum of amplitudes-squared does not equal one.")

            if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
                # Initialize the entire device state with the input state
                self.syncH2D(self._reshape(state, output_shape))
                return

            # generate basis states on subset of qubits via the cartesian product
            basis_states = np.array(list(product([0, 1], repeat=len(device_wires))))

            # get basis states to alter on full set of qubits
            unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
            unravelled_indices[:, device_wires] = basis_states

            # get indices for which the state is changed to input state vector elements
            ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

            self._kokkos_state.setStateVector(ravelled_indices, state)  # this operation on device

        def _apply_basis_state_kokkos(self, state, wires):
            """Initialize the state vector in a specified computational basis state.
            Args:
                state (array[int]): computational basis state of shape ``(wires,)``
                    consisting of 0s and 1s.
                wires (Wires): wires that the provided computational state should be initialized on
            Note: This function does not support broadcasted inputs yet.
            """
            # translate to wire labels used by device
            device_wires = self.map_wires(wires)

            # length of basis state parameter
            n_basis_state = len(state)

            if not set(state.tolist()).issubset({0, 1}):
                raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

            if n_basis_state != len(device_wires):
                raise ValueError("BasisState parameter and wires must be of equal length.")

            # get computational basis state number
            basis_states = 2 ** (self.num_wires - 1 - np.array(device_wires))
            basis_states = qml.math.convert_like(basis_states, state)
            num = int(qml.math.dot(state, basis_states))

            self._create_basis_state_kokkos(num)

        # To be able to validate the adjoint method [_validate_adjoint_method(device)],
        #  the qnode requires the definition of:
        # ["_apply_operation", "_apply_unitary", "adjoint_jacobian"]
        def _apply_operation():
            pass

        def _apply_unitary():
            pass

        def apply_kokkos(self, operations, **kwargs):
            # Skip over identity operations instead of performing
            # matrix multiplication with the identity.
            skipped_ops = ["Identity"]
            invert_param = False

            for o in operations:
                if str(o.name) in skipped_ops:
                    continue
                name = o.name
                if isinstance(o, Adjoint):
                    name = o.base.name
                    invert_param = True
                method = getattr(self._kokkos_state, name, None)

                wires = self.wires.indices(o.wires)

                if method is None:
                    # Inverse can be set to False since qml.matrix(o) is already in inverted form
                    try:
                        mat = qml.matrix(o)
                    except AttributeError:  # pragma: no cover
                        # To support older versions of PL
                        mat = o.matrix

                    if len(mat) == 0:
                        raise Exception("Unsupported operation")
                    self._kokkos_state.apply(
                        name,
                        wires,
                        False,
                        [],
                        mat.ravel(order="C"),  # inv = False: Matrix already in correct form;
                    )  # Parameters can be ignored for explicit matrices; F-order for cuQuantum

                else:
                    param = o.parameters
                    method(wires, invert_param, param)

        def apply(self, operations, **kwargs):
            # State preparation is currently done in Python
            if operations:  # make sure operations[0] exists
                if isinstance(operations[0], QubitStateVector):
                    self._apply_state_vector_kokkos(
                        operations[0].parameters[0].copy(), operations[0].wires
                    )
                    del operations[0]
                elif isinstance(operations[0], BasisState):
                    self._apply_basis_state_kokkos(operations[0].parameters[0], operations[0].wires)
                    del operations[0]

            for operation in operations:
                if isinstance(operation, (QubitStateVector, BasisState)):
                    raise DeviceError(
                        f"Operation {operation.name} cannot be used after other Operations have already been applied on a {self.short_name} device."
                    )

            self.apply_kokkos(operations)

        def generate_samples(self):
            """Generate samples

            Returns:
                array[int]: array of samples in binary representation with shape ``(dev.shots, dev.num_wires)``
            """
            return self._kokkos_state.GenerateSamples(len(self.wires), self.shots).astype(int)

        def var(self, observable, shot_range=None, bin_size=None):
            if self.shots is not None:
                # estimate the var
                # Lightning doesn't support sampling yet
                samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
                return np.squeeze(np.var(samples, axis=0))

            adjoint_matrix = math.T(math.conj(qml.matrix(observable)))
            sqr_matrix = np.matmul(adjoint_matrix, qml.matrix(observable))

            mean = self._kokkos_state.ExpectationValue(
                [f"{i}_var" for i in observable.name],
                self.wires.indices(observable.wires),
                observable.parameters,
                qml.matrix(observable).ravel(order="C"),
            )

            squared_mean = self._kokkos_state.ExpectationValue(
                [f"{i}_sqr" for i in observable.name],
                self.wires.indices(observable.wires),
                observable.parameters,
                sqr_matrix.ravel(order="C"),
            )

            return squared_mean - (mean**2)

        def probability(self, wires=None, shot_range=None, bin_size=None):
            """Return the probability of each computational basis state.

            Devices that require a finite number of shots always return the
            estimated probability.

            Args:
                wires (Iterable[Number, str], Number, str, Wires): wires to return
                    marginal probabilities for. Wires not provided are traced out of the system.
                shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                    to use. If not specified, all samples are used.
                bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                    returns the measurement statistic separately over each bin. If not
                    provided, the entire shot range is treated as a single bin.

            Returns:
                array[float]: list of the probabilities
            """

            if self.shots is not None:
                return self.estimate_probability(
                    wires=wires, shot_range=shot_range, bin_size=bin_size
                )

            wires = wires or self.wires
            wires = Wires(wires)

            # translate to wire labels used by device
            device_wires = self.map_wires(wires)

            if (
                device_wires
                and len(device_wires) > 1
                and (not np.all(np.array(device_wires)[:-1] <= np.array(device_wires)[1:]))
            ):
                raise RuntimeError(
                    "Lightning does not currently support out-of-order indices for probabilities"
                )

            return self._kokkos_state.probs(device_wires)

        def sample(self, observable, shot_range=None, bin_size=None, counts=False):
            if observable.name != "PauliZ":
                self.apply_kokkos(observable.diagonalizing_gates())
                self._samples = self.generate_samples()
            return super().sample(
                observable, shot_range=shot_range, bin_size=bin_size, counts=counts
            )

        def expval(self, observable, shot_range=None, bin_size=None):
            if observable.name in [
                "Projector",
            ]:
                return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

            if observable.name in ["SparseHamiltonian"]:
                CSR_SparseHamiltonian = observable.sparse_matrix().tocsr()
                return self._kokkos_state.ExpectationValue(
                    CSR_SparseHamiltonian.data,
                    CSR_SparseHamiltonian.indices,
                    CSR_SparseHamiltonian.indptr,
                )

            if observable.name in ["Hamiltonian"]:
                if len(observable.wires) < 13:
                    device_wires = self.map_wires(observable.wires)
                    return self._kokkos_state.ExpectationValue(
                        device_wires, qml.matrix(observable).ravel(order="C")
                    )
                else:
                    Hmat = qml.utils.sparse_hamiltonian(observable, wires=self.wires)
                    CSR_SparseHamiltonian = observable.sparse_matrix().tocsr()
                    return self._kokkos_state.ExpectationValue(
                        CSR_SparseHamiltonian.data,
                        CSR_SparseHamiltonian.indices,
                        CSR_SparseHamiltonian.indptr,
                    )

            if self.shots is not None:
                # estimate the expectation value
                samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
                return np.squeeze(np.mean(samples, axis=0))

            par = (
                observable.parameters
                if (
                    len(observable.parameters) > 0
                    and isinstance(observable.parameters[0], np.floating)
                )
                else []
            )

            if isinstance(observable.name, list):
                terms = observable.obs
                terms.sort(key=lambda x: self.wires.indices(x.wires))

            return self._kokkos_state.ExpectationValue(
                observable.name,
                self.wires.indices(observable.wires),
                par,  # observables should not pass parameters, use matrix instead
                qml.matrix(observable).ravel(order="C"),
            )

        @staticmethod
        def _check_adjdiff_supported_measurements(
            measurements: List[MeasurementProcess],
        ):
            """Check whether given list of measurement is supported by adjoint_diff.
            Args:
                measurements (List[MeasurementProcess]): a list of measurement processes to check.
            Returns:
                Expectation or State: a common return type of measurements.
            """
            if not measurements:
                return None

            if len(measurements) == 1 and measurements[0].return_type is State:
                # return State
                raise QuantumFunctionError("Not supported")

            # The return_type of measurement processes must be expectation
            if any(m.return_type is not Expectation for m in measurements):
                raise QuantumFunctionError(
                    "Adjoint differentiation method does not support expectation return type "
                    "mixed with other return types"
                )

            for m in measurements:
                if not isinstance(m.obs, Tensor):
                    if isinstance(m.obs, Projector):
                        raise QuantumFunctionError(
                            "Adjoint differentiation method does not support the Projector observable"
                        )
                    if isinstance(m.obs, Hermitian):
                        raise QuantumFunctionError(
                            "LightningKokkos adjoint differentiation method does not currently support the Hermitian observable"
                        )
                else:
                    if any(isinstance(o, Projector) for o in m.obs.non_identity_obs):
                        raise QuantumFunctionError(
                            "Adjoint differentiation method does not support the Projector observable"
                        )
                    if any(isinstance(o, Hermitian) for o in m.obs.non_identity_obs):
                        raise QuantumFunctionError(
                            "LightningKokkos adjoint differentiation method does not currently support the Hermitian observable"
                        )
            return Expectation

        @staticmethod
        def _check_adjdiff_supported_operations(operations):
            """Check Lightning adjoint differentiation method support for a tape.
            Args:
                tape (.QuantumTape): quantum tape to differentiate.
            Raise:
                QuantumFunctionError: if ``tape`` contains not supported measurements, observables, or operations by the Lightning adjoint differentiation method.
            """
            for op in operations:
                if op.num_params > 1 and not isinstance(op, Rot):
                    raise QuantumFunctionError(
                        f"The {op.name} operation is not supported using "
                        'the "adjoint" differentiation method'
                    )

        def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False, **kwargs):
            if self.shots is not None:
                warn(
                    "Requested adjoint differentiation to be computed with finite shots."
                    " The derivative is always exact when using the adjoint differentiation method.",
                    UserWarning,
                )

            tape_return_type = self._check_adjdiff_supported_measurements(tape.measurements)

            if len(tape.trainable_params) == 0:
                return np.array(0)

            # Check adjoint diff support
            self._check_adjdiff_supported_operations(tape.operations)

            # Initialization of state
            if starting_state is not None:
                ket = np.ravel(starting_state, order="C")
            elif not use_device_state:
                self.reset()
                self.execute(tape)

            if self.use_csingle:
                adj = AdjointJacobianKokkos_C64()
                ket = ket.astype(np.complex64)
            else:
                adj = AdjointJacobianKokkos_C128()

            obs_serialized = _serialize_observables(
                tape, self.wire_map, use_csingle=self.use_csingle
            )
            ops_serialized, use_sp = _serialize_ops(
                tape, self.wire_map, use_csingle=self.use_csingle
            )

            ops_serialized = adj.create_ops_list(*ops_serialized)

            trainable_params = sorted(tape.trainable_params)

            tp_shift = []
            record_tp_rows = []
            all_params = 0

            for op_idx, tp in enumerate(trainable_params):
                # get op_idx-th operator among differentiable operators
                op, _, _ = tape.get_operation(op_idx)

                if isinstance(op, Operation) and not isinstance(op, (BasisState, QubitStateVector)):
                    # We now just ignore non-op or state preps
                    tp_shift.append(tp)
                    record_tp_rows.append(all_params)
                all_params += 1

            if use_sp:
                # When the first element of the tape is state preparation. Still, I am not sure
                # whether there must be only one state preparation...
                tp_shift = [i - 1 for i in tp_shift]

            jac = adj.adjoint_jacobian(self._kokkos_state, obs_serialized, ops_serialized, tp_shift)
            jac = np.array(jac)  # only for parameters differentiable with the adjoint method
            jac = jac.reshape(-1, len(tp_shift))
            jac_r = np.zeros((jac.shape[0], all_params))
            jac_r[:, record_tp_rows] = jac
            return self._adjoint_jacobian_processing(jac_r) if active_return() else jac_r

        def vjp(self, measurements, dy, starting_state=None, use_device_state=False):
            """Generate the processing function required to compute the vector-Jacobian products of a tape."""
            if self.shots is not None:
                warn(
                    "Requested adjoint differentiation to be computed with finite shots."
                    " The derivative is always exact when using the adjoint differentiation method.",
                    UserWarning,
                )

            tape_return_type = self._check_adjdiff_supported_measurements(measurements)

            if math.allclose(dy, 0) or tape_return_type is None:
                return lambda tape: math.convert_like(np.zeros(len(tape.trainable_params)), dy)

            if tape_return_type is Expectation:
                if len(dy) != len(measurements):
                    raise ValueError(
                        "Number of observables in the tape must be the same as the length of dy in the vjp method"
                    )

                if np.iscomplexobj(dy):
                    raise ValueError(
                        "The vjp method only works with a real-valued dy when the tape is returning an expectation value"
                    )

                ham = qml.Hamiltonian(dy, [m.obs for m in measurements])

                def processing_fn(tape):
                    nonlocal ham
                    num_params = len(tape.trainable_params)

                    if num_params == 0:
                        return np.array([], dtype=self._state.dtype)

                    new_tape = tape.copy()
                    new_tape._measurements = [qml.expval(ham)]

                    return self.adjoint_jacobian(new_tape, starting_state, use_device_state)

                return processing_fn

        def _get_diagonalizing_gates(self, circuit: qml.tape.QuantumTape) -> List[Operation]:
            skip_diagonalizing = lambda obs: isinstance(obs, qml.Hamiltonian) or (
                isinstance(obs, qml.ops.Sum) and obs._pauli_rep is not None
            )
            meas_filtered = list(
                filter(
                    lambda m: m.obs is None or not skip_diagonalizing(m.obs),
                    circuit.measurements,
                )
            )
            return super()._get_diagonalizing_gates(
                qml.tape.QuantumScript(measurements=meas_filtered)
            )

else:  # CPP_BINARY_AVAILABLE:

    class LightningKokkos(LightningQubit):
        name = "PennyLane plugin for Kokkos-backed Lightning device"
        short_name = "lightning.kokkos"
        pennylane_requires = ">=0.30"
        version = __version__
        author = "Xanadu Inc."
        _CPP_BINARY_AVAILABLE = False

        def __init__(self, wires, *, c_dtype=np.complex128, **kwargs):
            w_msg = """
            !!!#####################################################################################
            !!!
            !!! WARNING: INSUFFICIENT SUPPORT DETECTED FOR KOKKOS DEVICE WITH `lightning.kokkos`
            !!!          DEFAULTING TO CPU DEVICE `lightning.qubit`
            !!!
            !!!#####################################################################################
            """
            warn(
                w_msg,
                RuntimeWarning,
            )
            super().__init__(wires, c_dtype=c_dtype, **kwargs)
