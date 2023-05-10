// Copyright 2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file StateVectorKokkos.hpp
 */

#pragma once
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "Error.hpp"
#include "GateFunctors.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Lightning_Kokkos::Util;
using namespace Pennylane::Functors;
} // namespace
/// @endcond

namespace Pennylane {

/**
 * @brief Kokkos functor for initializing the state vector to the \f$\ket{0}\f$
 * state
 *
 * @tparam Precision Floating point precision of underlying statevector data
 */
template <typename Precision> struct InitView {
    Kokkos::View<Kokkos::complex<Precision> *> a;
    InitView(Kokkos::View<Kokkos::complex<Precision> *> a_) : a(a_) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i) const {
        a(i) = Kokkos::complex<Precision>((i == 0) * 1.0, 0.0);
    }
};

/**
 * @brief Kokkos functor for setting the basis state
 *
 * @tparam Precision Floating point precision of underlying statevector data
 */
template <typename Precision> struct setBasisStateFunctor {
    Kokkos::View<Kokkos::complex<Precision> *> a;
    const std::size_t index;
    setBasisStateFunctor(Kokkos::View<Kokkos::complex<Precision> *> a_,
                         const std::size_t index_)
        : a(a_), index(index_) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i) const {
        a(i) = Kokkos::complex<Precision>((i == index) * 1.0, 0.0);
    }
};

/**
 * @brief Kokkos functor for setting the state vector
 *
 * @tparam Precision Floating point precision of underlying statevector data
 */
template <typename Precision> struct setStateVectorFunctor {
    Kokkos::View<Kokkos::complex<Precision> *> a;
    Kokkos::View<size_t *> indices;
    Kokkos::View<Kokkos::complex<Precision> *> values;
    setStateVectorFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> a_,
        const Kokkos::View<size_t *> indices_,
        const Kokkos::View<Kokkos::complex<Precision> *> values_)
        : a(a_), indices(indices_), values(values_) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i) const { a(indices[i]) = values[i]; }
};

/**
 * @brief Kokkos functor for initializing zeros to the state vector.
 *
 * @tparam Precision Floating point precision of underlying statevector data
 */
template <typename Precision> struct initZerosFunctor {
    Kokkos::View<Kokkos::complex<Precision> *> a;
    initZerosFunctor(Kokkos::View<Kokkos::complex<Precision> *> a_) : a(a_) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i) const {
        a(i) = Kokkos::complex<Precision>(0.0, 0.0);
    }
};

/**
 * @brief  Kokkos state vector class
 *
 * @tparam Precision Floating-point precision type.
 */
template <class Precision> class StateVectorKokkos {

  public:
    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    using KokkosVector = Kokkos::View<Kokkos::complex<Precision> *>;
    using KokkosSizeTVector = Kokkos::View<size_t *>;
    using KokkosRangePolicy = Kokkos::RangePolicy<KokkosExecSpace>;
    using UnmanagedComplexHostView =
        Kokkos::View<Kokkos::complex<Precision> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedSizeTHostView =
        Kokkos::View<size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstComplexHostView =
        Kokkos::View<const Kokkos::complex<Precision> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstSizeTHostView =
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    StateVectorKokkos() = delete;
    StateVectorKokkos(size_t num_qubits, const Kokkos::InitializationSettings &kokkos_args = {})
        : gates_{
                //Identity
                 {"PauliX", 
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyPauliX(std::forward<decltype(wires)>(wires),
                                  std::forward<decltype(adjoint)>(adjoint),
                                  std::forward<decltype(params)>(params));
                  }},
                 {"PauliY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyPauliY(std::forward<decltype(wires)>(wires),
                                  std::forward<decltype(adjoint)>(adjoint),
                                  std::forward<decltype(params)>(params));
                  }},
                 {"PauliZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyPauliZ(std::forward<decltype(wires)>(wires),
                                  std::forward<decltype(adjoint)>(adjoint),
                                  std::forward<decltype(params)>(params));
                  }},
                 {"Hadamard",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyHadamard(std::forward<decltype(wires)>(wires),
                                    std::forward<decltype(adjoint)>(adjoint),
                                    std::forward<decltype(params)>(params));
                  }},
                 {"S",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyS(std::forward<decltype(wires)>(wires),
                             std::forward<decltype(adjoint)>(adjoint),
                             std::forward<decltype(params)>(params));
                 }},
                 {"T",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyT(std::forward<decltype(wires)>(wires),
                             std::forward<decltype(adjoint)>(adjoint),
                             std::forward<decltype(params)>(params));
                  }},
                 {"RX",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyRX(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"RY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyRY(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"RZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyRZ(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"PhaseShift",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyPhaseShift(std::forward<decltype(wires)>(wires),
                                      std::forward<decltype(adjoint)>(adjoint),
                                      std::forward<decltype(params)>(params));
                  }},
                 {"Rot",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyRot(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params)>(params));
                  }},
                 {"CY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCY(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"CZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCZ(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"CNOT",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCNOT(std::forward<decltype(wires)>(wires),
                                std::forward<decltype(adjoint)>(adjoint),
                                std::forward<decltype(params)>(params));
                  }},
                 {"SWAP",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applySWAP(std::forward<decltype(wires)>(wires),
                                std::forward<decltype(adjoint)>(adjoint),
                                std::forward<decltype(params)>(params));
                  }},
                 {"ControlledPhaseShift",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyControlledPhaseShift(
                          std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params)>(params));
                  }},
                 {"CRX",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCRX(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params)>(params));
                  }},
                 {"CRY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCRY(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params)>(params));
                  }},
                 {"CRZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCRZ(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params)>(params));
                  }},
                 {"CRot",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCRot(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params)>(params));
                  }},                  
                  {"IsingXX",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyIsingXX(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},
                  {"IsingXY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyIsingXY(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},

                  {"IsingYY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyIsingYY(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},

                  {"IsingZZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyIsingZZ(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},
                 {"SingleExcitation",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applySingleExcitation(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},
                 {"SingleExcitationMinus",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applySingleExcitationMinus(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},
                 {"SingleExcitationPlus",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applySingleExcitationPlus(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},
                 {"DoubleExcitation",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyDoubleExcitation(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},
                 {"DoubleExcitationMinus",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyDoubleExcitationMinus(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},
                 {"DoubleExcitationPlus",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyDoubleExcitationPlus(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},
                 {"MultiRZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyMultiRZ(std::forward<decltype(wires)>(wires),
                                 std::forward<decltype(adjoint)>(adjoint),
                                 std::forward<decltype(params)>(params));
                  }},
                 {"CSWAP",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCSWAP(std::forward<decltype(wires)>(wires),
                                 std::forward<decltype(adjoint)>(adjoint),
                                 std::forward<decltype(params)>(params));
                  }},
                 {"Toffoli",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyToffoli(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }}
                 },
            generator_{
                {"RX",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorRX(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                {"RY",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorRY(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                {"RZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorRZ(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
		{"ControlledPhaseShift",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      return applyGeneratorControlledPhaseShift(
                          std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params)>(params));
                  }},
                {"CRX",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorCRX(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"CRY",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorCRY(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"CRZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorCRZ(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"IsingXX",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorIsingXX(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"IsingXY",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorIsingXY(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"IsingYY",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorIsingYY(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"IsingZZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorIsingZZ(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"SingleExcitation",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorSingleExcitation(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"SingleExcitationMinus",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorSingleExcitationMinus(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"SingleExcitationPlus",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorSingleExcitationPlus(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"DoubleExcitation",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorDoubleExcitation(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"DoubleExcitationMinus",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorDoubleExcitationMinus(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"DoubleExcitationPlus",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorDoubleExcitationPlus(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"PhaseShift",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorPhaseShift(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                  {"MultiRZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params){
                      return applyGeneratorMultiRZ(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
            }
    {
        num_qubits_ = num_qubits;
        length_ = Pennylane::Lightning_Kokkos::Util::exp2(num_qubits);

        {
            const std::lock_guard<std::mutex> lock(init_mutex_);
            if (!Kokkos::is_initialized()) {
                Kokkos::initialize(kokkos_args);
            }
        }

        if (num_qubits > 0) {
            data_ = std::make_unique<KokkosVector>(
                "data_", Lightning_Kokkos::Util::exp2(num_qubits));
            Kokkos::parallel_for(length_, InitView(*data_));
        }
    };

    /**
     * @brief Init zeros for the state-vector on device.
     */
    void initZeros() {
        Kokkos::parallel_for(getLength(), initZerosFunctor(getData()));
    }

    /**
     * @brief Set value for a single element of the state-vector on device.
     *
     * @param index Index of the target element.
     */
    void setBasisState(const size_t index) {
        Kokkos::parallel_for(getLength(),
                             setBasisStateFunctor(getData(), index));
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param values Values to be set for the target elements.
     * @param indices Indices of the target elements.
     */
    void setStateVector(const std::vector<std::size_t> &indices,
                        const std::vector<Kokkos::complex<Precision>> &values) {

        initZeros();

        KokkosSizeTVector d_indices("d_indices", indices.size());

        KokkosVector d_values("d_values", values.size());

        Kokkos::deep_copy(d_indices, UnmanagedConstSizeTHostView(
                                         indices.data(), indices.size()));

        Kokkos::deep_copy(d_values, UnmanagedConstComplexHostView(
                                        values.data(), values.size()));

        Kokkos::parallel_for(
            indices.size(),
            setStateVectorFunctor(getData(), d_indices, d_values));
    }

    /**
     * @brief Reset the data back to the \f$\ket{0}\f$ state.
     *
     * @param num_qubits Number of qubits
     */
    void resetStateVector() {
        if (length_ > 0) {
            Kokkos::parallel_for(length_, InitView(*data_));
        }
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkos(Kokkos::complex<Precision> *hostdata_, size_t length,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(Lightning_Kokkos::Util::log2(length), kokkos_args) {
        HostToDevice(hostdata_, length);
    }

    /**
     * @brief Copy constructor
     *
     * @param other Another state vector
     */
    StateVectorKokkos(const StateVectorKokkos &other,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(other.getNumQubits(), kokkos_args) {
        this->DeviceToDevice(other.getData());
    }

    /**
     * @brief Destructor for StateVectorKokkos class
     *
     * @param other Another state vector
     */
    ~StateVectorKokkos() {
        data_.reset();
        {
            const std::lock_guard<std::mutex> lock(init_mutex_);
            if (!is_exit_reg_) {
                is_exit_reg_ = true;
                std::atexit([]() {
                    if (!Kokkos::is_finalized()) {
                        Kokkos::finalize();
                    }
                });
            }
        }
    }

    /**
     * @brief Apply a single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional gate matrix if opName doesn't exist
     */
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &wires, bool adjoint = false,
                        const std::vector<Precision> &params = {0.0},
                        [[maybe_unused]] const KokkosVector &gate_matrix = {}) {
        if (opName == "Identity") {
            // No op
        } else if (gates_.find(opName) != gates_.end()) {
            gates_.at(opName)(wires, adjoint, params);
        } else {
            KokkosVector matrix("gate_matrix", gate_matrix.size());
            Kokkos::deep_copy(matrix,
                              UnmanagedComplexHostView(gate_matrix.data(),
                                                       gate_matrix.size()));
            return applyMultiQubitOp(matrix, wires, adjoint);
        }
    }

    /**
     * @brief Apply a single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param params Optional std gate matrix if opName doesn't exist.
     */
    void applyOperation_std(
        const std::string &opName, const std::vector<size_t> &wires,
        bool adjoint = false, const std::vector<Precision> &params = {0.0},
        [[maybe_unused]] const std::vector<Kokkos::complex<Precision>>
            &gate_matrix = {}) {

        if (opName == "Identity") {
            // No op
        } else if (gates_.find(opName) != gates_.end()) {
            gates_.at(opName)(wires, adjoint, params);
        } else {
            KokkosVector matrix("gate_matrix", gate_matrix.size());
            Kokkos::deep_copy(
                matrix, UnmanagedConstComplexHostView(gate_matrix.data(),
                                                      gate_matrix.size()));
            return applyMultiQubitOp(matrix, wires, adjoint);
        }
    }

    /**
     * @brief Multi-op variant of execute(const std::string &opName, const
     std::vector<int> &wires, bool adjoint = false, const std::vector<Precision>
     &params)
     *
     * @param opNames Name of gates to apply.
     * @param wires Wires to apply gate to.
     * @param adjoints Indicates whether to use adjoint of gate.
     * @param params parameter list for parametric gates.
     */
    void applyOperation(const std::vector<std::string> &opNames,
                        const std::vector<std::vector<size_t>> &wires,
                        const std::vector<bool> &adjoints,
                        const std::vector<std::vector<Precision>> &params) {
        PL_ABORT_IF(opNames.size() != wires.size(),
                    "Incompatible number of ops and wires");
        PL_ABORT_IF(opNames.size() != adjoints.size(),
                    "Incompatible number of ops and adjoints");
        const auto num_ops = opNames.size();
        for (std::size_t op_idx = 0; op_idx < num_ops; op_idx++) {
            applyOperation(opNames[op_idx], wires[op_idx], adjoints[op_idx],
                           params[op_idx]);
        }
    }

    /**
     * @brief Multi-op variant of execute(const std::string &opName, const
     std::vector<int> &wires, bool adjoint = false, const std::vector<Precision>
     &params)
     *
     * @param opNames Name of gates to apply.
     * @param wires Wires to apply gate to.
     * @param adjoints Indicates whether to use adjoint of gate.
     * @param params parameter list for parametric gates.
     */
    void applyOperation(const std::vector<std::string> &opNames,
                        const std::vector<std::vector<size_t>> &wires,
                        const std::vector<bool> &adjoints) {
        PL_ABORT_IF(opNames.size() != wires.size(),
                    "Incompatible number of ops and wires");
        PL_ABORT_IF(opNames.size() != adjoints.size(),
                    "Incompatible number of ops and adjoints");
        const auto num_ops = opNames.size();
        for (std::size_t op_idx = 0; op_idx < num_ops; op_idx++) {
            applyOperation(opNames[op_idx], wires[op_idx], adjoints[op_idx]);
        }
    }

    /**
     * @brief Apply a single generator to the state vector using the given
     * kernel.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     */
    auto applyGenerator(const std::string &opName,
                        const std::vector<size_t> &wires, bool adjoint = false,
                        const std::vector<Precision> &params = {0.0})
        -> Precision {
        const auto it = generator_.find(opName);
        PL_ABORT_IF(it == generator_.end(),
                    std::string("Generator does not exist for ") + opName);
        return (it->second)(wires, adjoint, params);
    }

    /**
     * @brief Apply a single qubit operator to the state vector using a matrix
     *
     * @param matrix Kokkos gate matrix in the device space
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     */
    void applySingleQubitOp(const KokkosVector &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        auto &&num_qubits = getNumQubits();
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Lightning_Kokkos::Util::exp2(num_qubits - 1)),
                singleQubitOpFunctor<Precision, false>(*data_, num_qubits,
                                                       matrix, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Lightning_Kokkos::Util::exp2(num_qubits - 1)),
                singleQubitOpFunctor<Precision, true>(*data_, num_qubits,
                                                      matrix, wires));
        }
    }

    /**
     * @brief Apply a two qubit operator to the state vector using a matrix
     *
     * @param matrix Kokkos gate matrix in the device space
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     */
    void applyTwoQubitOp(const KokkosVector &matrix,
                         const std::vector<size_t> &wires,
                         bool inverse = false) {

        auto &&num_qubits = getNumQubits();
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Lightning_Kokkos::Util::exp2(num_qubits - 2)),
                twoQubitOpFunctor<Precision, false>(*data_, num_qubits, matrix,
                                                    wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Lightning_Kokkos::Util::exp2(num_qubits - 2)),
                twoQubitOpFunctor<Precision, true>(*data_, num_qubits, matrix,
                                                   wires));
        }
    }

    /**
     * @brief Apply a multi qubit operator to the state vector using a matrix
     *
     * @param matrix Kokkos gate matrix in the device space
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     */
    void applyMultiQubitOp(const KokkosVector &matrix,
                           const std::vector<size_t> &wires,
                           bool inverse = false) {
        auto &&num_qubits = getNumQubits();
        if (wires.size() == 1) {
            applySingleQubitOp(matrix, wires, inverse);
        } else if (wires.size() == 2) {
            applyTwoQubitOp(matrix, wires, inverse);
        } else {

            Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                wires_host(wires.data(), wires.size());

            Kokkos::View<std::size_t *> wires_view("wires_view", wires.size());
            Kokkos::deep_copy(wires_view, wires_host);

            if (!inverse) {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<KokkosExecSpace>(
                        0, Lightning_Kokkos::Util::exp2(num_qubits_ -
                                                        wires.size())),
                    multiQubitOpFunctor<Precision, false>(*data_, num_qubits,
                                                          matrix, wires_view));
            } else {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<KokkosExecSpace>(
                        0, Lightning_Kokkos::Util::exp2(num_qubits_ -
                                                        wires.size())),
                    multiQubitOpFunctor<Precision, true>(*data_, num_qubits,
                                                         matrix, wires_view));
            }
        }
    }

    /**
     * @brief Templated method that applies special n-qubit gates.
     *
     * @tparam functor_t Gate functor class for Kokkos dispatcher.
     * @tparam nqubits Number of qubits.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    template <template <class, bool> class functor_t, int nqubits>
    void applyGateFunctor(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {}) {
        auto &&num_qubits = getNumQubits();
        PL_ASSERT(wires.size() == nqubits);
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Lightning_Kokkos::Util::exp2(num_qubits - nqubits)),
                functor_t<Precision, false>(*data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Lightning_Kokkos::Util::exp2(num_qubits - nqubits)),
                functor_t<Precision, true>(*data_, num_qubits, wires, params));
        }
    }

    /**
     * @brief Apply a PauliX operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Parameters for this gate
     */
    void
    applyPauliX(const std::vector<size_t> &wires, bool inverse = false,
                [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<pauliXFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a PauliY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Parameters for this gate
     */
    void
    applyPauliY(const std::vector<size_t> &wires, bool inverse = false,
                [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<pauliYFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a PauliZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Parameters for this gate
     */

    void
    applyPauliZ(const std::vector<size_t> &wires, bool inverse = false,
                [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<pauliZFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a Hadamard operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void
    applyHadamard(const std::vector<size_t> &wires, bool inverse = false,
                  [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<hadamardFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a S operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyS(const std::vector<size_t> &wires, bool inverse = false,
                [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<sFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a T operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyT(const std::vector<size_t> &wires, bool inverse = false,
                [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<tFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a RX operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyRX(const std::vector<size_t> &wires, bool inverse,
                 [[maybe_unused]] const std::vector<Precision> &params) {
        applyGateFunctor<rxFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a RY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyRY(const std::vector<size_t> &wires, bool inverse,
                 [[maybe_unused]] const std::vector<Precision> &params) {
        applyGateFunctor<ryFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a RZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyRZ(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<rzFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a PhaseShift operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyPhaseShift(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<phaseShiftFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a Rot operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyRot(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Precision> &params) {
        applyGateFunctor<rotFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a CY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCY(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<cyFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a CZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCZ(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<czFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a CNOT operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCNOT(const std::vector<size_t> &wires, bool inverse = false,
                   [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<cnotFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a SWAP operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applySWAP(const std::vector<size_t> &wires, bool inverse = false,
                   [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<swapFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a ControlledPhaseShift operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyControlledPhaseShift(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<controlledPhaseShiftFunctor, 2>(wires, inverse,
                                                         params);
    }

    /**
     * @brief Apply a CRX operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCRX(const std::vector<size_t> &wires, bool inverse = false,
                  [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<crxFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a CRY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCRY(const std::vector<size_t> &wires, bool inverse = false,
                  [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<cryFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a CRZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCRZ(const std::vector<size_t> &wires, bool inverse = false,
                  [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<crzFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a CRot operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCRot(const std::vector<size_t> &wires, bool inverse,
                   [[maybe_unused]] const std::vector<Precision> &params) {
        applyGateFunctor<cRotFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a IsingXX operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void
    applyIsingXX(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<isingXXFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a IsingXY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void
    applyIsingXY(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<isingXYFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a IsingYY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void
    applyIsingYY(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<isingYYFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a IsingZZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void
    applyIsingZZ(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<isingZZFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a SingleExcitation operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applySingleExcitation(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<singleExcitationFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a SingleExcitationMinus operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applySingleExcitationMinus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<singleExcitationMinusFunctor, 2>(wires, inverse,
                                                          params);
    }

    /**
     * @brief Apply a SingleExcitationPlus operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applySingleExcitationPlus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<singleExcitationPlusFunctor, 2>(wires, inverse,
                                                         params);
    }

    /**
     * @brief Apply a DoubleExcitation operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyDoubleExcitation(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<doubleExcitationFunctor, 4>(wires, inverse, params);
    }

    /**
     * @brief Apply a DoubleExcitationMinus operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyDoubleExcitationMinus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<doubleExcitationMinusFunctor, 4>(wires, inverse,
                                                          params);
    }

    /**
     * @brief Apply a DoubleExcitationPlus operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyDoubleExcitationPlus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<doubleExcitationPlusFunctor, 4>(wires, inverse,
                                                         params);
    }

    /**
     * @brief Apply a MultiRZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void
    applyMultiRZ(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<Precision> &params = {}) {
        auto &&num_qubits = getNumQubits();

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Lightning_Kokkos::Util::exp2(num_qubits)),
                multiRZFunctor<Precision, false>(*data_, num_qubits, wires,
                                                 params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Lightning_Kokkos::Util::exp2(num_qubits)),
                multiRZFunctor<Precision, true>(*data_, num_qubits, wires,
                                                params));
        }
    }

    /**
     * @brief Apply a CSWAP operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void
    applyCSWAP(const std::vector<size_t> &wires, bool inverse = false,
               [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<cSWAPFunctor, 3>(wires, inverse, params);
    }

    /**
     * @brief Apply a Toffoli operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void
    applyToffoli(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<Precision> &params = {}) {
        applyGateFunctor<toffoliFunctor, 3>(wires, inverse, params);
    }

    /**
     * @brief Apply a PhaseShift generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorPhaseShift(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorPhaseShiftFunctor, 1>(wires, inverse, params);
        return static_cast<Precision>(1.0);
    }

    /**
     * @brief Apply a IsingXX generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorIsingXX(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorIsingXXFunctor, 2>(wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a IsingXY generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorIsingXY(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorIsingXYFunctor, 2>(wires, inverse, params);
        return static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a IsingYY generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorIsingYY(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorIsingYYFunctor, 2>(wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a IsingZZ generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorIsingZZ(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorIsingZZFunctor, 2>(wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a SingleExcitation generator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorSingleExcitation(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorSingleExcitationFunctor, 2>(wires, inverse,
                                                              params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a SingleExcitationMinus generator to the state vector using
     * a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorSingleExcitationMinus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorSingleExcitationMinusFunctor, 2>(
            wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a SingleExcitationPlus generator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorSingleExcitationPlus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorSingleExcitationPlusFunctor, 2>(
            wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a DoubleExcitation generator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorDoubleExcitation(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorDoubleExcitationFunctor, 4>(wires, inverse,
                                                              params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a DoubleExcitationMinus generator to the state vector using
     * a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorDoubleExcitationMinus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorDoubleExcitationMinusFunctor, 4>(
            wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a DoubleExcitationPlus generator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorDoubleExcitationPlus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorDoubleExcitationPlusFunctor, 4>(
            wires, inverse, params);
        return static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a RX generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto
    applyGeneratorRX(const std::vector<size_t> &wires, bool inverse = false,
                     [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyPauliX(wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a RY generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto
    applyGeneratorRY(const std::vector<size_t> &wires, bool inverse = false,
                     [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyPauliY(wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a RZ generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto
    applyGeneratorRZ(const std::vector<size_t> &wires, bool inverse = false,
                     [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyPauliZ(wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a ControlledPhaseShift generator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorControlledPhaseShift(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorControlledPhaseShiftFunctor, 2>(
            wires, inverse, params);
        return static_cast<Precision>(1);
    }

    /**
     * @brief Apply a CRX generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorCRX(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})

        -> Precision {
        applyGateFunctor<generatorCRXFunctor, 2>(wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a CRY generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorCRY(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorCRYFunctor, 2>(wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a CRZ generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorCRZ(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        applyGateFunctor<generatorCRZFunctor, 2>(wires, inverse, params);
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Apply a MultiRZ generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorMultiRZ(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<Precision> &params = {})
        -> Precision {
        auto &&num_qubits = getNumQubits();

        if (inverse == false) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Lightning_Kokkos::Util::exp2(num_qubits)),
                generatorMultiRZFunctor<Precision, false>(*data_, num_qubits,
                                                          wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Lightning_Kokkos::Util::exp2(num_qubits)),
                generatorMultiRZFunctor<Precision, true>(*data_, num_qubits,
                                                         wires));
        }
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Get the number of qubits of the state vector.
     *
     * @return The number of qubits of the state vector
     */
    size_t getNumQubits() const { return num_qubits_; }

    /**
     * @brief Get the size of the state vector
     *
     * @return The size of the state vector
     */
    size_t getLength() const { return length_; }

    void updateData(const StateVectorKokkos<Precision> &other) {
        Kokkos::deep_copy(*data_, other.getData());
    }

    /**
     * @brief Get the Kokkos data of the state vector.
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getData() const -> KokkosVector & { return *data_; }

    /**
     * @brief Get the Kokkos data of the state vector
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getData() -> KokkosVector & { return *data_; }

    /**
     * @brief Copy data from the host space to the device space.
     *
     */
    inline void HostToDevice(Kokkos::complex<Precision> *sv, size_t length) {
        Kokkos::deep_copy(*data_, UnmanagedComplexHostView(sv, length));
    }

    /**
     * @brief Copy data from the device space to the host space.
     *
     */
    inline void DeviceToHost(Kokkos::complex<Precision> *sv, size_t length) {
        Kokkos::deep_copy(UnmanagedComplexHostView(sv, length), *data_);
    }

    /**
     * @brief Copy data from the device space to the device space.
     *
     */
    inline void DeviceToDevice(KokkosVector vector_to_copy) {
        Kokkos::deep_copy(*data_, vector_to_copy);
    }

  private:
    using GateFunc = std::function<void(const std::vector<size_t> &, bool,
                                        const std::vector<Precision> &)>;
    using GateMap = std::unordered_map<std::string, GateFunc>;
    const GateMap gates_;

    using GeneratorFunc = std::function<Precision(
        const std::vector<size_t> &, bool, const std::vector<Precision> &)>;
    using GeneratorMap = std::unordered_map<std::string, GeneratorFunc>;
    const GeneratorMap generator_;

    size_t num_qubits_;
    size_t length_;
    std::mutex init_mutex_;
    std::unique_ptr<KokkosVector> data_;
    inline static bool is_exit_reg_ = false;
};

}; // namespace Pennylane
