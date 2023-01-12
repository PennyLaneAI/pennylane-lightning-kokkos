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
#include "Bindings.hpp"
#include "Error.hpp" // LightningException
#include "StateVectorKokkos.hpp"

#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Algorithms;
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
        .def(py::init([](const np_arr_c &arr) {
            py::buffer_info numpyArrayInfo = arr.request();
            auto *data_ptr =
                static_cast<Kokkos::complex<PrecisionT> *>(numpyArrayInfo.ptr);
            return new StateVectorKokkos<PrecisionT>(
                data_ptr, static_cast<std::size_t>(arr.size()));
        }))
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
                return sv.getExpectationValue(obsName, wires, params,
                                              conv_matrix);
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
                return sv.getExpectationValue(
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
                return sv.getExpectationValue(wires, conv_matrix);
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
                return sv.getExpectationValue(conv_data, indices, index_ptr);
            },
            "Calculate the expectation value of the given observable.")
        .def("probs",
             [](StateVectorKokkos<PrecisionT> &sv,
                const std::vector<size_t> &wires) {
                 if (wires.empty()) {
                     return py::array_t<ParamT>(py::cast(sv.probs()));
                 }

                 const bool is_sorted_wires =
                     std::is_sorted(wires.begin(), wires.end());

                 if (wires.size() == sv.getNumQubits()) {
                     if (is_sorted_wires)
                         return py::array_t<ParamT>(py::cast(sv.probs()));
                 }
                 return py::array_t<ParamT>(py::cast(sv.probs(wires)));
             })
        .def("GenerateSamples",
             [](StateVectorKokkos<PrecisionT> &sv, size_t num_wires,
                size_t num_shots) {
                 auto &&result = sv.generate_samples(num_shots);

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

    class_name = "ObsStructKokkos_C" + bitsize;
    using obs_data_var = std::variant<std::monostate, np_arr_r, np_arr_c>;
    py::class_<ObsDatum<PrecisionT>>(m, class_name.c_str(), py::module_local())
        .def(py::init([](const std::vector<std::string> &names,
                         const std::vector<obs_data_var> &params,
                         const std::vector<std::vector<size_t>> &wires) {
            std::vector<typename ObsDatum<PrecisionT>::param_var_t> conv_params(
                params.size());
            for (size_t p_idx = 0; p_idx < params.size(); p_idx++) {
                std::visit(
                    [&](const auto &param) {
                        using p_t = std::decay_t<decltype(param)>;
                        if constexpr (std::is_same_v<p_t, np_arr_c>) {
                            auto buffer = param.request();
                            auto ptr =
                                static_cast<std::complex<ParamT> *>(buffer.ptr);
                            if (buffer.size) {
                                conv_params[p_idx] =
                                    std::vector<std::complex<ParamT>>{
                                        ptr, ptr + buffer.size};
                            }
                        } else if constexpr (std::is_same_v<p_t, np_arr_r>) {
                            auto buffer = param.request();

                            auto *ptr = static_cast<ParamT *>(buffer.ptr);
                            if (buffer.size) {
                                conv_params[p_idx] =
                                    std::vector<ParamT>{ptr, ptr + buffer.size};
                            }
                        } else {
                            PL_ABORT(
                                "Parameter datatype not current supported");
                        }
                    },
                    params[p_idx]);
            }
            return ObsDatum<PrecisionT>(names, conv_params, wires);
        }))
        .def("__repr__",
             [](const ObsDatum<PrecisionT> &obs) {
                 using namespace Pennylane::Util;
                 std::ostringstream obs_stream;
                 std::string obs_name = obs.getObsName()[0];
                 for (size_t o = 1; o < obs.getObsName().size(); o++) {
                     if (o < obs.getObsName().size()) {
                         obs_name += " @ ";
                     }
                     obs_name += obs.getObsName()[o];
                 }
                 obs_stream << "'wires' : " << obs.getObsWires();
                 return "Observable: { 'name' : " + obs_name + ", " +
                        obs_stream.str() + " }";
             })
        .def("get_name",
             [](const ObsDatum<PrecisionT> &obs) { return obs.getObsName(); })
        .def("get_wires",
             [](const ObsDatum<PrecisionT> &obs) { return obs.getObsWires(); })
        .def("get_params", [](const ObsDatum<PrecisionT> &obs) {
            py::list params;
            for (size_t i = 0; i < obs.getObsParams().size(); i++) {
                std::visit(
                    [&](const auto &param) {
                        using p_t = std::decay_t<decltype(param)>;
                        if constexpr (std::is_same_v<
                                          p_t,
                                          std::vector<std::complex<ParamT>>>) {
                            params.append(py::array_t<std::complex<ParamT>>(
                                py::cast(param)));
                        } else if constexpr (std::is_same_v<
                                                 p_t, std::vector<ParamT>>) {
                            params.append(py::array_t<ParamT>(py::cast(param)));
                        } else if constexpr (std::is_same_v<p_t,
                                                            std::monostate>) {
                            params.append(py::list{});
                        } else {
                            throw("Unsupported data type");
                        }
                    },
                    obs.getObsParams()[i]);
            }
            return params;
        });

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
            using namespace Pennylane::Util;
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
                const std::vector<Pennylane::Algorithms::ObsDatum<PrecisionT>>
                    &observables,
                const Pennylane::Algorithms::OpsData<PrecisionT> &operations,
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
    m.def("kokkos_config", &getConfig, "Kokkos configurations query.");
}
}

} // namespace
  /// @endcond
