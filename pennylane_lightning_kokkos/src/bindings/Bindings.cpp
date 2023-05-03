// Copyright 2018-2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <set>
#include <tuple>
#include <variant>
#include <vector>

#include "AdjointDiffKokkos.hpp"
#include "Error.hpp"         // LightningException
#include "GetConfigInfo.hpp" // Kokkos configuration info
#include "MeasuresKokkos.hpp"
#include "StateVectorKokkos.hpp"

#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Lightning_Kokkos::Algorithms;
using namespace Pennylane::Lightning_Kokkos::Simulators;
using std::complex;
using std::set;
using std::string;
using std::vector;

namespace py = pybind11;

/**
 * @brief Templated class to build all required precisions for Python module.
 *
 * @tparam PrecisionT Precision of the statevector data.
 * @tparam ParamT Precision of the parameter data.
 * @param m Pybind11 module.
 */
template <class PrecisionT, class ParamT>
void StateVectorKokkos_class_bindings(py::module &m) {

    using np_arr_r =
        py::array_t<ParamT, py::array::c_style | py::array::forcecast>;
    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;

    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);
    std::string class_name = "LightningKokkos_C" + bitsize;

    py::class_<StateVectorKokkos<PrecisionT>>(m, class_name.c_str())
        .def(py::init([](std::size_t num_qubits) {
            return new StateVectorKokkos<PrecisionT>(num_qubits);
        }))
        .def(py::init([](std::size_t num_qubits,
                         const Kokkos::InitializationSettings &kokkos_args) {
            return new StateVectorKokkos<PrecisionT>(num_qubits, kokkos_args);
        }))
        .def(py::init([](const np_arr_c &arr) {
            py::buffer_info numpyArrayInfo = arr.request();
            auto *data_ptr =
                static_cast<Kokkos::complex<PrecisionT> *>(numpyArrayInfo.ptr);
            return new StateVectorKokkos<PrecisionT>(
                data_ptr, static_cast<std::size_t>(arr.size()));
        }))
        .def(py::init([](const np_arr_c &arr,
                         const Kokkos::InitializationSettings &kokkos_args) {
            py::buffer_info numpyArrayInfo = arr.request();
            auto *data_ptr =
                static_cast<Kokkos::complex<PrecisionT> *>(numpyArrayInfo.ptr);
            return new StateVectorKokkos<PrecisionT>(
                data_ptr, static_cast<std::size_t>(arr.size()), kokkos_args);
        }))
        .def(
            "setBasisState",
            [](StateVectorKokkos<PrecisionT> &sv, const size_t index) {
                sv.setBasisState(index);
            },
            "Create Basis State on Device.")
        .def(
            "setStateVector",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &indices, const np_arr_c &state) {
                const auto buffer = state.request();
                std::vector<Kokkos::complex<ParamT>> state_kok;
                if (buffer.size) {
                    const auto ptr =
                        static_cast<const Kokkos::complex<ParamT> *>(
                            buffer.ptr);
                    state_kok = std::vector<Kokkos::complex<ParamT>>{
                        ptr, ptr + buffer.size};
                }
                sv.setStateVector(indices, state_kok);
            },
            "Set State Vector on device with values and their corresponding "
            "indices for the state vector on device")
        .def(
            "Identity",
            []([[maybe_unused]] StateVectorKokkos<PrecisionT> &sv,
               [[maybe_unused]] const std::vector<std::size_t> &wires,
               [[maybe_unused]] bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {},
            "Apply the Identity gate.")
        .def(
            "PauliX",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applyPauliX(wires, adjoint);
            },
            "Apply the PauliX gate.")

        .def(
            "PauliY",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applyPauliY(wires, adjoint);
            },
            "Apply the PauliY gate.")

        .def(
            "PauliZ",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applyPauliZ(wires, adjoint);
            },
            "Apply the PauliZ gate.")

        .def(
            "Hadamard",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applyHadamard(wires, adjoint);
            },
            "Apply the Hadamard gate.")

        .def(
            "S",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applyS(wires, adjoint);
            },
            "Apply the S gate.")

        .def(
            "T",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applyT(wires, adjoint);
            },
            "Apply the T gate.")

        .def(
            "CNOT",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applyCNOT(wires, adjoint);
            },
            "Apply the CNOT gate.")

        .def(
            "SWAP",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applySWAP(wires, adjoint);
            },
            "Apply the SWAP gate.")

        .def(
            "CSWAP",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applyCSWAP(wires, adjoint);
            },
            "Apply the CSWAP gate.")

        .def(
            "Toffoli",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applyToffoli(wires, adjoint);
            },
            "Apply the Toffoli gate.")

        .def(
            "CY",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applyCY(wires, adjoint);
            },
            "Apply the CY gate.")

        .def(
            "CZ",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                sv.applyCZ(wires, adjoint);
            },
            "Apply the CZ gate.")

        .def(
            "PhaseShift",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                sv.applyPhaseShift(wires, adjoint, params);
            },
            "Apply the PhaseShift gate.")

        .def("apply",
             py::overload_cast<
                 const vector<string> &, const vector<vector<std::size_t>> &,
                 const vector<bool> &, const vector<vector<PrecisionT>> &>(
                 &StateVectorKokkos<PrecisionT>::applyOperation))

        .def("apply", py::overload_cast<const vector<string> &,
                                        const vector<vector<std::size_t>> &,
                                        const vector<bool> &>(
                          &StateVectorKokkos<PrecisionT>::applyOperation))
        .def(
            "apply",
            [](StateVectorKokkos<PrecisionT> &sv, const std::string &str,
               const vector<size_t> &wires, bool inv,
               [[maybe_unused]] const std::vector<std::vector<ParamT>> &params,
               [[maybe_unused]] const np_arr_c &gate_matrix) {
                const auto m_buffer = gate_matrix.request();
                std::vector<Kokkos::complex<ParamT>> conv_matrix;
                if (m_buffer.size) {
                    const auto m_ptr =
                        static_cast<const Kokkos::complex<ParamT> *>(
                            m_buffer.ptr);
                    conv_matrix = std::vector<Kokkos::complex<ParamT>>{
                        m_ptr, m_ptr + m_buffer.size};
                }
                sv.applyOperation_std(str, wires, inv, std::vector<ParamT>{},
                                      conv_matrix);
            },
            "Apply operation via the gate matrix")
        .def("applyGenerator",
             py::overload_cast<const std::string &, const std::vector<size_t> &,
                               bool, const vector<PrecisionT> &>(
                 &StateVectorKokkos<PrecisionT>::applyGenerator))
        .def(
            "ControlledPhaseShift",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                sv.applyControlledPhaseShift(wires, adjoint, params);
            },
            "Apply the ControlledPhaseShift gate.")

        .def(
            "RX",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                sv.applyRX(wires, adjoint, params);
            },
            "Apply the RX gate.")

        .def(
            "RY",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                sv.applyRY(wires, adjoint, params);
            },
            "Apply the RY gate.")

        .def(
            "RZ",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                sv.applyRZ(wires, adjoint, params);
            },
            "Apply the RZ gate.")

        .def(
            "Rot",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                sv.applyRot(wires, adjoint, params);
            },
            "Apply the Rot gate.")

        .def(
            "CRX",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                sv.applyCRX(wires, adjoint, params);
            },
            "Apply the CRX gate.")

        .def(
            "CRY",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                sv.applyCRY(wires, adjoint, params);
            },
            "Apply the CRY gate.")

        .def(
            "CRZ",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                sv.applyCRZ(wires, adjoint, params);
            },
            "Apply the CRZ gate.")

        .def(
            "CRot",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                return sv.applyCRot(wires, adjoint, params);
            },
            "Apply the CRot gate.")
        .def(
            "IsingXX",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                return sv.applyIsingXX(wires, adjoint, params);
            },
            "Apply the IsingXX gate.")
        .def(
            "IsingXY",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                sv.applyIsingXY(wires, adjoint, params);
            },
            "Apply the IsingXY gate.")
        .def(
            "IsingYY",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                return sv.applyIsingYY(wires, adjoint, params);
            },
            "Apply the IsingYY gate.")
        .def(
            "IsingZZ",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                return sv.applyIsingZZ(wires, adjoint, params);
            },
            "Apply the IsingZZ gate.")
        .def(
            "MultiRZ",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                return sv.applyMultiRZ(wires, adjoint, params);
            },
            "Apply the MultiRZ gate.")
        .def(
            "SingleExcitation",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                return sv.applySingleExcitation(wires, adjoint, params);
            },
            "Apply the SingleExcitation gate.")
        .def(
            "SingleExcitationMinus",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                return sv.applySingleExcitationMinus(wires, adjoint, params);
            },
            "Apply the SingleExcitationMinus gate.")
        .def(
            "SingleExcitationPlus",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                return sv.applySingleExcitationPlus(wires, adjoint, params);
            },
            "Apply the SingleExcitationPlus gate.")
        .def(
            "DoubleExcitation",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                return sv.applyDoubleExcitation(wires, adjoint, params);
            },
            "Apply the DoubleExcitation gate.")
        .def(
            "DoubleExcitationMinus",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                return sv.applyDoubleExcitationMinus(wires, adjoint, params);
            },
            "Apply the DoubleExcitationMinus gate.")
        .def(
            "DoubleExcitationPlus",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires, bool adjoint,
               const std::vector<ParamT> &params) {
                return sv.applyDoubleExcitationPlus(wires, adjoint, params);
            },
            "Apply the DoubleExcitationPlus gate.")
        .def(
            "ExpectationValue",
            [](StateVectorKokkos<PrecisionT> &sv, const std::string &obsName,
               const std::vector<std::size_t> &wires,
               [[maybe_unused]] const std::vector<ParamT> &params,
               [[maybe_unused]] const np_arr_c &gate_matrix) {
                const auto m_buffer = gate_matrix.request();
                std::vector<Kokkos::complex<ParamT>> conv_matrix;
                if (m_buffer.size) {
                    auto m_ptr =
                        static_cast<Kokkos::complex<ParamT> *>(m_buffer.ptr);
                    conv_matrix = std::vector<Kokkos::complex<ParamT>>{
                        m_ptr, m_ptr + m_buffer.size};
                }
                // Return the real component only
                return MeasuresKokkos<PrecisionT>(sv).getExpectationValue(
                    obsName, wires, params, conv_matrix);
            },
            "Calculate the expectation value of the given observable.")
        .def(
            "ExpectationValue",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::string> &obsName,
               const std::vector<std::size_t> &wires,
               [[maybe_unused]] const std::vector<std::vector<ParamT>> &params,
               [[maybe_unused]] const np_arr_c &gate_matrix) {
                std::string obs_concat{"#"};
                for (const auto &sub : obsName) {
                    obs_concat += sub;
                }
                const auto m_buffer = gate_matrix.request();
                std::vector<Kokkos::complex<ParamT>> conv_matrix;
                if (m_buffer.size) {
                    const auto m_ptr =
                        static_cast<const Kokkos::complex<ParamT> *>(
                            m_buffer.ptr);
                    conv_matrix = std::vector<Kokkos::complex<ParamT>>{
                        m_ptr, m_ptr + m_buffer.size};
                }
                // Return the real component only & ignore params
                return MeasuresKokkos<PrecisionT>(sv).getExpectationValue(
                    obs_concat, wires, std::vector<ParamT>{}, conv_matrix);
            },
            "Calculate the expectation value of the given observable.")
        .def(
            "ExpectationValue",
            [](StateVectorKokkos<PrecisionT> &sv,
               const std::vector<std::size_t> &wires,
               const np_arr_c &gate_matrix) {
                const auto m_buffer = gate_matrix.request();
                std::vector<Kokkos::complex<ParamT>> conv_matrix;
                if (m_buffer.size) {
                    const auto m_ptr =
                        static_cast<const Kokkos::complex<ParamT> *>(
                            m_buffer.ptr);
                    conv_matrix = std::vector<Kokkos::complex<ParamT>>{
                        m_ptr, m_ptr + m_buffer.size};
                }
                // Return the real component only & ignore params
                return MeasuresKokkos<PrecisionT>(sv).getExpectationValue(
                    wires, conv_matrix);
            },
            "Calculate the expectation value of the given observable.")

        .def(
            "ExpectationValue",
            [](StateVectorKokkos<PrecisionT> &sv, const np_arr_c &gate_data,
               const std::vector<std::size_t> &indices,
               const std::vector<std::size_t> &index_ptr) {
                const auto m_buffer = gate_data.request();
                std::vector<Kokkos::complex<ParamT>> conv_data;
                if (m_buffer.size) {
                    const auto m_ptr =
                        static_cast<const Kokkos::complex<ParamT> *>(
                            m_buffer.ptr);
                    conv_data = std::vector<Kokkos::complex<ParamT>>{
                        m_ptr, m_ptr + m_buffer.size};
                }
                // Return the real component only & ignore params
                return MeasuresKokkos<PrecisionT>(sv).getExpectationValue(
                    conv_data, indices, index_ptr);
            },
            "Calculate the expectation value of the given observable.")
        .def("probs",
             [](StateVectorKokkos<PrecisionT> &sv,
                const std::vector<size_t> &wires) {
                 auto m = MeasuresKokkos<PrecisionT>(sv);
                 if (wires.empty()) {
                     return py::array_t<ParamT>(py::cast(m.probs()));
                 }

                 const bool is_sorted_wires =
                     std::is_sorted(wires.begin(), wires.end());

                 if (wires.size() == sv.getNumQubits()) {
                     if (is_sorted_wires)
                         return py::array_t<ParamT>(py::cast(m.probs()));
                 }
                 return py::array_t<ParamT>(py::cast(m.probs(wires)));
             })
        .def("GenerateSamples",
             [](StateVectorKokkos<PrecisionT> &sv, size_t num_wires,
                size_t num_shots) {
                 auto &&result =
                     MeasuresKokkos<PrecisionT>(sv).generate_samples(num_shots);

                 const size_t ndim = 2;
                 const std::vector<size_t> shape{num_shots, num_wires};
                 constexpr auto sz = sizeof(size_t);
                 const std::vector<size_t> strides{sz * num_wires, sz};
                 // return 2-D NumPy array
                 return py::array(py::buffer_info(
                     result.data(), /* data as contiguous array  */
                     sz,            /* size of one scalar        */
                     py::format_descriptor<size_t>::format(), /* data type */
                     ndim,   /* number of dimensions      */
                     shape,  /* shape of the matrix       */
                     strides /* strides for each axis     */
                     ));
             })
        .def(
            "DeviceToHost",
            [](StateVectorKokkos<PrecisionT> &device_sv, np_arr_c &host_sv) {
                py::buffer_info numpyArrayInfo = host_sv.request();
                auto *data_ptr = static_cast<Kokkos::complex<PrecisionT> *>(
                    numpyArrayInfo.ptr);
                if (host_sv.size()) {
                    device_sv.DeviceToHost(data_ptr, host_sv.size());
                }
            },
            "Synchronize data from the GPU device to host.")
        .def("HostToDevice",
             py::overload_cast<Kokkos::complex<PrecisionT> *, size_t>(
                 &StateVectorKokkos<PrecisionT>::HostToDevice),
             "Synchronize data from the host device to GPU.")
        .def(
            "HostToDevice",
            [](StateVectorKokkos<PrecisionT> &device_sv,
               const np_arr_c &host_sv) {
                const py::buffer_info numpyArrayInfo = host_sv.request();
                auto *data_ptr = static_cast<Kokkos::complex<PrecisionT> *>(
                    numpyArrayInfo.ptr);
                const auto length =
                    static_cast<size_t>(numpyArrayInfo.shape[0]);
                if (length) {
                    device_sv.HostToDevice(data_ptr, length);
                }
            },
            "Synchronize data from the host device to GPU.")
        .def("numQubits", &StateVectorKokkos<PrecisionT>::getNumQubits)
        .def("dataLength", &StateVectorKokkos<PrecisionT>::getLength)
        .def("resetKokkos", &StateVectorKokkos<PrecisionT>::resetStateVector);

    //***********************************************************************//
    //                              Observable
    //***********************************************************************//

    class_name = "ObservableKokkos_C" + bitsize;
    py::class_<ObservableKokkos<PrecisionT>,
               std::shared_ptr<ObservableKokkos<PrecisionT>>>(
        m, class_name.c_str(), py::module_local());

    class_name = "NamedObsKokkos_C" + bitsize;
    py::class_<NamedObsKokkos<PrecisionT>,
               std::shared_ptr<NamedObsKokkos<PrecisionT>>,
               ObservableKokkos<PrecisionT>>(m, class_name.c_str(),
                                             py::module_local())
        .def(py::init(
            [](const std::string &name, const std::vector<size_t> &wires) {
                return NamedObsKokkos<PrecisionT>(name, wires);
            }))
        .def("__repr__", &NamedObsKokkos<PrecisionT>::getObsName)
        .def("get_wires", &NamedObsKokkos<PrecisionT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const NamedObsKokkos<PrecisionT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<NamedObsKokkos<PrecisionT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<NamedObsKokkos<PrecisionT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HermitianObsKokkos_C" + bitsize;
    py::class_<HermitianObsKokkos<PrecisionT>,
               std::shared_ptr<HermitianObsKokkos<PrecisionT>>,
               ObservableKokkos<PrecisionT>>(m, class_name.c_str(),
                                             py::module_local())
        .def(py::init([](const np_arr_c &matrix,
                         const std::vector<size_t> &wires) {
            const auto m_buffer = matrix.request();
            std::vector<std::complex<ParamT>> conv_matrix;
            if (m_buffer.size) {
                const auto m_ptr =
                    static_cast<const std::complex<PrecisionT> *>(m_buffer.ptr);
                conv_matrix = std::vector<std::complex<PrecisionT>>{
                    m_ptr, m_ptr + m_buffer.size};
            }
            return HermitianObsKokkos<PrecisionT>(conv_matrix, wires);
        }))
        .def("__repr__", &HermitianObsKokkos<PrecisionT>::getObsName)
        .def("get_wires", &HermitianObsKokkos<PrecisionT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const HermitianObsKokkos<PrecisionT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<HermitianObsKokkos<PrecisionT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<HermitianObsKokkos<PrecisionT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "TensorProdObsKokkos_C" + bitsize;
    py::class_<TensorProdObsKokkos<PrecisionT>,
               std::shared_ptr<TensorProdObsKokkos<PrecisionT>>,
               ObservableKokkos<PrecisionT>>(m, class_name.c_str(),
                                             py::module_local())
        .def(py::init(
            [](const std::vector<std::shared_ptr<ObservableKokkos<PrecisionT>>>
                   &obs) { return TensorProdObsKokkos<PrecisionT>(obs); }))
        .def("__repr__", &TensorProdObsKokkos<PrecisionT>::getObsName)
        .def("get_wires", &TensorProdObsKokkos<PrecisionT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const TensorProdObsKokkos<PrecisionT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<TensorProdObsKokkos<PrecisionT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<TensorProdObsKokkos<PrecisionT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HamiltonianKokkos_C" + bitsize;
    using ObsPtr = std::shared_ptr<ObservableKokkos<PrecisionT>>;
    py::class_<HamiltonianKokkos<PrecisionT>,
               std::shared_ptr<HamiltonianKokkos<PrecisionT>>,
               ObservableKokkos<PrecisionT>>(m, class_name.c_str(),
                                             py::module_local())
        .def(py::init(
            [](const np_arr_r &coeffs, const std::vector<ObsPtr> &obs) {
                auto buffer = coeffs.request();
                const auto ptr = static_cast<const ParamT *>(buffer.ptr);
                return HamiltonianKokkos<PrecisionT>{
                    std::vector(ptr, ptr + buffer.size), obs};
            }))
        .def("__repr__", &HamiltonianKokkos<PrecisionT>::getObsName)
        .def("get_wires", &HamiltonianKokkos<PrecisionT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const HamiltonianKokkos<PrecisionT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<HamiltonianKokkos<PrecisionT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<HamiltonianKokkos<PrecisionT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "SparseHamiltonianKokkos_C" + bitsize;
    py::class_<SparseHamiltonianKokkos<PrecisionT>,
               std::shared_ptr<SparseHamiltonianKokkos<PrecisionT>>,
               ObservableKokkos<PrecisionT>>(m, class_name.c_str(),
                                             py::module_local())
        .def(py::init([](const np_arr_c &data,
                         const std::vector<std::size_t> &indices,
                         const std::vector<std::size_t> &indptr,
                         const std::vector<std::size_t> &wires) {
            const py::buffer_info buffer_data = data.request();
            const auto *data_ptr =
                static_cast<std::complex<PrecisionT> *>(buffer_data.ptr);

            return SparseHamiltonianKokkos<PrecisionT>{
                std::vector<std::complex<PrecisionT>>(
                    {data_ptr, data_ptr + data.size()}),
                indices, indptr, wires};
        }))
        .def("__repr__", &SparseHamiltonianKokkos<PrecisionT>::getObsName)
        .def("get_wires", &SparseHamiltonianKokkos<PrecisionT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const SparseHamiltonianKokkos<PrecisionT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<SparseHamiltonianKokkos<PrecisionT>>(
                        other)) {
                    return false;
                }
                auto other_cast =
                    other.cast<SparseHamiltonianKokkos<PrecisionT>>();
                return self == other_cast;
            },
            "Compare two observables");

    //***********************************************************************//
    //                              Operations
    //***********************************************************************//
    class_name = "OpsStructKokkos_C" + bitsize;
    py::class_<OpsData<PrecisionT>>(m, class_name.c_str(), py::module_local())
        .def(py::init<
             const std::vector<std::string> &,
             const std::vector<std::vector<ParamT>> &,
             const std::vector<std::vector<size_t>> &,
             const std::vector<bool> &,
             const std::vector<std::vector<std::complex<PrecisionT>>> &>())
        .def("__repr__", [](const OpsData<PrecisionT> &ops) {
            using namespace Pennylane::Lightning_Kokkos::Util;
            std::ostringstream ops_stream;
            for (size_t op = 0; op < ops.getSize(); op++) {
                ops_stream << "{'name': " << ops.getOpsName()[op];
                ops_stream << ", 'params': " << ops.getOpsParams()[op];
                ops_stream << ", 'inv': " << ops.getOpsInverses()[op];
                ops_stream << "}";
                if (op < ops.getSize() - 1) {
                    ops_stream << ",";
                }
            }
            return "Operations: [" + ops_stream.str() + "]";
        });

    //***********************************************************************//
    //                              Adj Jac
    //***********************************************************************//

    class_name = "AdjointJacobianKokkos_C" + bitsize;
    py::class_<AdjointJacobianKokkos<PrecisionT>>(m, class_name.c_str(),
                                                  py::module_local())
        .def(py::init<>())
        .def("create_ops_list",
             [](AdjointJacobianKokkos<PrecisionT> &adj,
                const std::vector<std::string> &ops_name,
                const std::vector<np_arr_r> &ops_params,
                const std::vector<std::vector<size_t>> &ops_wires,
                const std::vector<bool> &ops_inverses,
                const std::vector<np_arr_c> &ops_matrices) {
                 std::vector<std::vector<PrecisionT>> conv_params(
                     ops_params.size());
                 std::vector<std::vector<std::complex<PrecisionT>>>
                     conv_matrices(ops_matrices.size());
                 static_cast<void>(adj);
                 for (size_t op = 0; op < ops_name.size(); op++) {
                     const auto p_buffer = ops_params[op].request();
                     const auto m_buffer = ops_matrices[op].request();
                     if (p_buffer.size) {
                         const auto *const p_ptr =
                             static_cast<const ParamT *>(p_buffer.ptr);
                         conv_params[op] =
                             std::vector<ParamT>{p_ptr, p_ptr + p_buffer.size};
                     }

                     if (m_buffer.size) {
                         const auto m_ptr =
                             static_cast<const std::complex<ParamT> *>(
                                 m_buffer.ptr);
                         conv_matrices[op] = std::vector<std::complex<ParamT>>{
                             m_ptr, m_ptr + m_buffer.size};
                     }
                 }

                 return OpsData<PrecisionT>{ops_name, conv_params, ops_wires,
                                            ops_inverses, conv_matrices};
             })
        .def("adjoint_jacobian",
             &AdjointJacobianKokkos<PrecisionT>::adjointJacobian)
        .def("adjoint_jacobian",
             [](AdjointJacobianKokkos<PrecisionT> &adj,
                const StateVectorKokkos<PrecisionT> &sv,
                const std::vector<std::shared_ptr<ObservableKokkos<PrecisionT>>>
                    &observables,
                const OpsData<PrecisionT> &operations,
                const std::vector<size_t> &trainableParams) {
                 std::vector<std::vector<PrecisionT>> jac(
                     observables.size(),
                     std::vector<PrecisionT>(trainableParams.size(), 0));
                 adj.adjointJacobian(sv, jac, observables, operations,
                                     trainableParams, false);
                 return py::array_t<ParamT>(py::cast(jac));
             });
}

// Necessary to avoid mangled names when manually building module
// due to CUDA & LTO incompatibility issues.
extern "C" {
/**
 * @brief Add C++ classes, methods and functions to Python module.
 */
PYBIND11_MODULE(lightning_kokkos_qubit_ops, // NOLINT: No control over
                                            // Pybind internals
                m) {
    // Suppress doxygen autogenerated signatures

    py::options options;
    options.disable_function_signatures();
    py::register_exception<LightningException>(m, "PLException");

    StateVectorKokkos_class_bindings<float, float>(m);
    StateVectorKokkos_class_bindings<double, double>(m);

    m.def("kokkos_start", []() { Kokkos::initialize(); });
    m.def("kokkos_end", []() { Kokkos::finalize(); });
    m.def("kokkos_config_info", &getConfig, "Kokkos configurations query.");
    m.def(
        "print_configuration",
        []() {
            std::ostringstream buffer;
            Kokkos::print_configuration(buffer, true);
            return buffer.str();
        },
        "Kokkos configurations query.");

    py::class_<Kokkos::InitializationSettings>(m, "InitializationSettings")
        .def(py::init([]() {
            return Kokkos::InitializationSettings()
                .set_num_threads(0)
                .set_device_id(0)
                .set_map_device_id_by("")
                .set_disable_warnings(0)
                .set_print_configuration(0)
                .set_tune_internals(0)
                .set_tools_libs("")
                .set_tools_help(0)
                .set_tools_args("");
        }))
        .def("get_num_threads",
             &Kokkos::InitializationSettings::get_num_threads,
             "Number of threads to use with the host parallel backend. Must be "
             "greater than zero.")
        .def("get_device_id", &Kokkos::InitializationSettings::get_device_id,
             "Device to use with the device parallel backend. Valid IDs are "
             "zero to number of GPU(s) available for execution minus one.")
        .def(
            "get_map_device_id_by",
            &Kokkos::InitializationSettings::get_map_device_id_by,
            "Strategy to select a device automatically from the GPUs available "
            "for execution. Must be either mpi_rank"
            "for round-robin assignment based on the local MPI rank or random.")
        .def("get_disable_warnings",
             &Kokkos::InitializationSettings::get_disable_warnings,
             "Whether to disable warning messages.")
        .def("get_print_configuration",
             &Kokkos::InitializationSettings::get_print_configuration,
             "Whether to print the configuration after initialization.")
        .def("get_tune_internals",
             &Kokkos::InitializationSettings::get_tune_internals,
             "Whether to allow autotuning internals instead of using "
             "heuristics.")
        .def("get_tools_libs", &Kokkos::InitializationSettings::get_tools_libs,
             "Which tool dynamic library to load. Must either be the full path "
             "to library or the name of library if the path is present in the "
             "runtime library search path (e.g. LD_LIBRARY_PATH)")
        .def("get_tools_help", &Kokkos::InitializationSettings::get_tools_help,
             "Query the loaded tool for its command-line options support.")
        .def("get_tools_args", &Kokkos::InitializationSettings::get_tools_args,
             "Options to pass to the loaded tool as command-line arguments.")
        .def("has_num_threads",
             &Kokkos::InitializationSettings::has_num_threads,
             "Number of threads to use with the host parallel backend. Must be "
             "greater than zero.")
        .def("has_device_id", &Kokkos::InitializationSettings::has_device_id,
             "Device to use with the device parallel backend. Valid IDs are "
             "zero "
             "to number of GPU(s) available for execution minus one.")
        .def(
            "has_map_device_id_by",
            &Kokkos::InitializationSettings::has_map_device_id_by,
            "Strategy to select a device automatically from the GPUs available "
            "for execution. Must be either mpi_rank"
            "for round-robin assignment based on the local MPI rank or random.")
        .def("has_disable_warnings",
             &Kokkos::InitializationSettings::has_disable_warnings,
             "Whether to disable warning messages.")
        .def("has_print_configuration",
             &Kokkos::InitializationSettings::has_print_configuration,
             "Whether to print the configuration after initialization.")
        .def("has_tune_internals",
             &Kokkos::InitializationSettings::has_tune_internals,
             "Whether to allow autotuning internals instead of using "
             "heuristics.")
        .def("has_tools_libs", &Kokkos::InitializationSettings::has_tools_libs,
             "Which tool dynamic library to load. Must either be the full path "
             "to "
             "library or the name of library if the path is present in the "
             "runtime library search path (e.g. LD_LIBRARY_PATH)")
        .def("has_tools_help", &Kokkos::InitializationSettings::has_tools_help,
             "Query the loaded tool for its command-line options support.")
        .def("has_tools_args", &Kokkos::InitializationSettings::has_tools_args,
             "Options to pass to the loaded tool as command-line arguments.")
        .def("set_num_threads",
             &Kokkos::InitializationSettings::set_num_threads,
             "Number of threads to use with the host parallel backend. Must be "
             "greater than zero.")
        .def("set_device_id", &Kokkos::InitializationSettings::set_device_id,
             "Device to use with the device parallel backend. Valid IDs are "
             "zero to number of GPU(s) available for execution minus one.")
        .def(
            "set_map_device_id_by",
            &Kokkos::InitializationSettings::set_map_device_id_by,
            "Strategy to select a device automatically from the GPUs available "
            "for execution. Must be either mpi_rank"
            "for round-robin assignment based on the local MPI rank or random.")
        .def("set_disable_warnings",
             &Kokkos::InitializationSettings::set_disable_warnings,
             "Whether to disable warning messages.")
        .def("set_print_configuration",
             &Kokkos::InitializationSettings::set_print_configuration,
             "Whether to print the configuration after initialization.")
        .def("set_tune_internals",
             &Kokkos::InitializationSettings::set_tune_internals,
             "Whether to allow autotuning internals instead of using "
             "heuristics.")
        .def("set_tools_libs", &Kokkos::InitializationSettings::set_tools_libs,
             "Which tool dynamic library to load. Must either be the full path "
             "to library or the name of library if the path is present in the "
             "runtime library search path (e.g. LD_LIBRARY_PATH)")
        .def("set_tools_help", &Kokkos::InitializationSettings::set_tools_help,
             "Query the loaded tool for its command-line options support.")
        .def("set_tools_args", &Kokkos::InitializationSettings::set_tools_args,
             "Options to pass to the loaded tool as command-line arguments.")
        .def("__repr__", [](const Kokkos::InitializationSettings &args) {
            using namespace Pennylane::Lightning_Kokkos::Util;
            std::ostringstream args_stream;
            args_stream << args;
            return args_stream.str();
        });
}
}

} // namespace
  /// @endcond
