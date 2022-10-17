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
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "Error.hpp"
#include "ExpValFunctors.hpp"
#include "GateFunctors.hpp"
#include "MeasuresFunctors.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
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
 * @brief  Kokkos state vector class
 *
 * @tparam Precision Floating-point precision type.
 */
template <class Precision> class StateVectorKokkos {

  public:
    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    using KokkosVector = Kokkos::View<Kokkos::complex<Precision> *>;
    using KokkosRangePolicy = Kokkos::RangePolicy<KokkosExecSpace>;
    using UnmanagedComplexHostView =
        Kokkos::View<Kokkos::complex<Precision> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstComplexHostView =
        Kokkos::View<const Kokkos::complex<Precision> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    StateVectorKokkos() = delete;
    StateVectorKokkos(size_t num_qubits)
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

        expval_funcs_["Identity"] = [&](auto &&wires, auto &&params) {
            return getExpectationValueIdentity(
                std::forward<decltype(wires)>(wires),
                std::forward<decltype(params)>(params));
        };

        expval_funcs_["PauliX"] = [&](auto &&wires, auto &&params) {
            return getExpectationValuePauliX(
                std::forward<decltype(wires)>(wires),
                std::forward<decltype(params)>(params));
        };

        expval_funcs_["PauliY"] = [&](auto &&wires, auto &&params) {
            return getExpectationValuePauliY(
                std::forward<decltype(wires)>(wires),
                std::forward<decltype(params)>(params));
        };

        expval_funcs_["PauliZ"] = [&](auto &&wires, auto &&params) {
            return getExpectationValuePauliZ(
                std::forward<decltype(wires)>(wires),
                std::forward<decltype(params)>(params));
        };

        expval_funcs_["Hadamard"] = [&](auto &&wires, auto &&params) {
            return getExpectationValueHadamard(
                std::forward<decltype(wires)>(wires),
                std::forward<decltype(params)>(params));
        };

        num_qubits_ = num_qubits;
        length_ = Pennylane::Util::exp2(num_qubits);

        {
            const std::lock_guard<std::mutex> lock(counts_mutex_);
            if (counts_ == 0 and !Kokkos::is_initialized()) {
                Kokkos::initialize();
            }
            counts_++;
        }

        if (num_qubits > 0) {
            data_ =
                std::make_unique<KokkosVector>("data_", Util::exp2(num_qubits));
            Kokkos::parallel_for(length_, InitView(*data_));
        }
    };

    /**
     * @brief Utility method for samples.
     *
     * @param num_samples Number of Samples
     *
     * @return Kokkos::View<size_t *> to the samples.
     * Each sample has a length equal to the number of qubits. Each sample can
     * be accessed using the stride sample_id*num_qubits, where sample_id is a
     * number between 0 and num_samples-1.
     */

    auto generate_samples(size_t num_samples) -> std::vector<size_t> {

        const size_t num_qubits = getNumQubits();
        const size_t N = getLength();

        Kokkos::View<Kokkos::complex<Precision> *> arr_data = getData();
        Kokkos::View<Precision *> probabilities("probabilities", N);
        Kokkos::View<size_t *> samples("num_samples", num_samples * num_qubits);

        // Compute prob
        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(0, N),
            getProbFunctor<Precision>(arr_data, probabilities));

        Kokkos::parallel_scan(Kokkos::RangePolicy<KokkosExecSpace>(0, N),
                              getCDFFunctor<Precision>(probabilities));

        //  Sampling process
        Kokkos::Random_XorShift64_Pool<> rand_pool(5374857);

        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(0, num_samples),
            Sampler<Precision, Kokkos::Random_XorShift64_Pool>(
                samples, probabilities, rand_pool, num_qubits, N));

        std::vector<size_t> samples_h(num_samples * num_qubits);

        using UnmanagedSize_tHostView =
            Kokkos::View<size_t *, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        Kokkos::deep_copy(
            UnmanagedSize_tHostView(samples_h.data(), samples_h.size()),
            samples);

        return samples_h;
    }

    /**
     * @brief Reset the data back to the \f$\ket{0}\f$ state
     *
     * @param num_qubits Number of qubits
     */
    void resetStateVector() {
        if (length_ > 0) {
            Kokkos::parallel_for(length_, InitView(*data_));
        }
    }

    /**
     * @brief Create a new state vector from data on the host
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkos(Kokkos::complex<Precision> *hostdata_, size_t length)
        : StateVectorKokkos(Util::log2(length)) {
        HostToDevice(hostdata_, length);
    }

    /**
     * @brief Copy constructor
     *
     * @param other Another state vector
     */
    StateVectorKokkos(const StateVectorKokkos &other)
        : StateVectorKokkos(other.getNumQubits()) {
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
            const std::lock_guard<std::mutex> lock(counts_mutex_);
            counts_--;
            if (counts_ == 0) {
                Kokkos::finalize();
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
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applySingleQubitOpFunctor<Precision, false>(
                                     *data_, num_qubits, matrix, wires));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applySingleQubitOpFunctor<Precision, true>(
                                     *data_, num_qubits, matrix, wires));
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
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyTwoQubitOpFunctor<Precision, false>(
                                     *data_, num_qubits, matrix, wires));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyTwoQubitOpFunctor<Precision, true>(
                                     *data_, num_qubits, matrix, wires));
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
                        0, Util::exp2(num_qubits_ - wires.size())),
                    applyMultiQubitOpFunctor<Precision, false>(
                        *data_, num_qubits, matrix, wires_view));
            } else {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<KokkosExecSpace>(
                        0, Util::exp2(num_qubits_ - wires.size())),
                    applyMultiQubitOpFunctor<Precision, true>(
                        *data_, num_qubits, matrix, wires_view));
            }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyPauliXFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applyPauliXFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyPauliYFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applyPauliYFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyPauliZFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applyPauliZFunctor<Precision, true>(*data_, num_qubits, wires));
        }
    }

    /**
     * @brief Calculate the expectation value of an observable
     *
     * @param obsName observable name
     * @param wires wires the observable acts on
     * @param params parameters for the observable
     * @param gate_matrix optional matrix
     */
    auto getExpectationValue(
        const std::string &obsName, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params = {0.0},
        const std::vector<Kokkos::complex<Precision>> &gate_matrix = {}) {

        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;
        auto &&local_wires =
            (gate_matrix.empty())
                ? wires
                : std::vector<size_t>{
                      wires.rbegin(),
                      wires.rend()}; // ensure wire indexing correctly preserved
                                     // for tensor-observables

        if (expval_funcs_.find(obsName) != expval_funcs_.end()) {
            return expval_funcs_.at(obsName)(local_wires, par);
        } else {
            KokkosVector matrix("gate_matrix", gate_matrix.size());
            Kokkos::deep_copy(
                matrix, UnmanagedConstComplexHostView(gate_matrix.data(),
                                                      gate_matrix.size()));
            return getExpectationValueMultiQubitOp(matrix, wires, par);
        }
    }

    /**
     * @brief Calculate expectation value with respect to identity observable on
     * specified wire. For normalised states this function will always return 1.
     *
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Squared norm of state.
     */
    auto getExpectationValueIdentity(
        const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params = {0.0}) {

        Precision expval = 0;
        Kokkos::parallel_reduce(
            Util::exp2(num_qubits_),
            getExpectationValueIdentityFunctor(*data_, num_qubits_, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to Pauli X observable on
     * specified wire.
     *
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to Pauli X applied to specified
     * wire.
     */
    auto getExpectationValuePauliX(
        const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params = {0.0}) {

        Precision expval = 0;
        Kokkos::parallel_reduce(
            Util::exp2(num_qubits_ - 1),
            getExpectationValuePauliXFunctor(*data_, num_qubits_, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to Pauli Y observable on
     * specified wire.
     *
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to Pauli Y applied to specified
     * wire.
     */
    auto getExpectationValuePauliY(
        const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params = {0.0}) {

        Precision expval = 0;
        Kokkos::parallel_reduce(
            Util::exp2(num_qubits_ - 1),
            getExpectationValuePauliYFunctor(*data_, num_qubits_, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to Pauli Z observable on
     * specified wire.
     *
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to Pauli Z applied to specified
     * wire.
     */
    auto getExpectationValuePauliZ(
        const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params = {0.0}) {

        Precision expval = 0;
        Kokkos::parallel_reduce(
            Util::exp2(num_qubits_ - 1),
            getExpectationValuePauliZFunctor(*data_, num_qubits_, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to Hadamard observable on
     * specified wire.
     *
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to Hadamard applied to specified
     * wire.
     */
    auto getExpectationValueHadamard(
        const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params = {0.0}) {

        Precision expval = 0;
        Kokkos::parallel_reduce(
            Util::exp2(num_qubits_ - 1),
            getExpectationValueHadamardFunctor(*data_, num_qubits_, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to single qubit
     * observable on specified wire.
     *
     * @param matrix Hermitian matrix representing observable to be used.
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to observable applied to specified
     * wire.
     */
    auto getExpectationValueSingleQubitOp(
        const KokkosVector &matrix, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params = {0.0}) {

        Precision expval = 0;
        Kokkos::parallel_reduce(Util::exp2(num_qubits_ - 1),
                                getExpectationValueSingleQubitOpFunctor(
                                    *data_, num_qubits_, matrix, wires),
                                expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to two qubit observable
     * on specified wires.
     *
     * @param matrix Hermitian matrix representing observable to be used.
     * @param wires Wires to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to observable applied to specified
     * wires.
     */
    auto getExpectationValueTwoQubitOp(
        const KokkosVector &matrix, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params = {0.0}) {

        Precision expval = 0;
        Kokkos::parallel_reduce(Util::exp2(num_qubits_ - 2),
                                getExpectationValueTwoQubitOpFunctor(
                                    *data_, num_qubits_, matrix, wires),
                                expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to multi qubit observable
     * on specified wires.
     *
     * @param matrix Hermitian matrix representing observable to be used.
     * @param wires Wires to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to observable applied to specified
     * wires.
     */
    auto getExpectationValueMultiQubitOp(
        const KokkosVector &matrix, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params = {0.0}) {
        if (wires.size() == 1) {
            return getExpectationValueSingleQubitOp(matrix, wires, params);
        } else if (wires.size() == 2) {
            return getExpectationValueTwoQubitOp(matrix, wires, params);
        } else {
            Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                wires_host(wires.data(), wires.size());

            Kokkos::View<std::size_t *> wires_view("wires_view", wires.size());
            Kokkos::deep_copy(wires_view, wires_host);
            Precision expval = 0;
            Kokkos::parallel_reduce(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits_ - wires.size())),
                getExpectationValueMultiQubitOpFunctor(*data_, num_qubits_,
                                                       matrix, wires_view),
                expval);
            return expval;
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyHadamardFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyHadamardFunctor<Precision, true>(
                                     *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applySFunctor<Precision, false>(*data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applySFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applyTFunctor<Precision, false>(*data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applyTFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyRXFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyRXFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyRYFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyRYFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyRZFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyRZFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyPhaseShiftFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params[0]));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyPhaseShiftFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params[0]));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyRotFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyRotFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
    }

    /**
     * @brief Apply a CY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCY(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<Precision> &param = {}) {
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyCYFunctor<Precision, false>(*data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyCYFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyCZFunctor<Precision, false>(*data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyCZFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyCNOTFunctor<Precision, false>(*data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyCNOTFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applySWAPFunctor<Precision, false>(*data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applySWAPFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyControlledPhaseShiftFunctor<Precision, false>(
                    *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyControlledPhaseShiftFunctor<Precision, true>(
                    *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyCRXFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyCRXFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyCRYFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyCRYFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyCRZFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyCRZFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyCRotFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyCRotFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyIsingXXFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyIsingXXFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyIsingXYFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyIsingXYFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);
        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyIsingYYFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyIsingYYFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyIsingZZFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyIsingZZFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applySingleExcitationFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applySingleExcitationFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applySingleExcitationMinusFunctor<Precision, false>(
                    *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applySingleExcitationMinusFunctor<Precision, true>(
                    *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applySingleExcitationPlusFunctor<Precision, false>(
                    *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applySingleExcitationPlusFunctor<Precision, true>(
                    *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 4);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 4)),
                                 applyDoubleExcitationFunctor<Precision, false>(
                                     *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 4)),
                                 applyDoubleExcitationFunctor<Precision, true>(
                                     *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 4);

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 4)),
                applyDoubleExcitationMinusFunctor<Precision, false>(
                    *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 4)),
                applyDoubleExcitationMinusFunctor<Precision, true>(
                    *data_, num_qubits, wires, params));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 4);

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 4)),
                applyDoubleExcitationPlusFunctor<Precision, false>(
                    *data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 4)),
                applyDoubleExcitationPlusFunctor<Precision, true>(
                    *data_, num_qubits, wires, params));
        }
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
                Kokkos::RangePolicy<KokkosExecSpace>(0, Util::exp2(num_qubits)),
                applyMultiRZFunctor<Precision, false>(*data_, num_qubits, wires,
                                                      params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, Util::exp2(num_qubits)),
                applyMultiRZFunctor<Precision, true>(*data_, num_qubits, wires,
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 3);

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 3)),
                applyCSWAPFunctor<Precision, false>(*data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 3)),
                applyCSWAPFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 3);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 3)),
                                 applyToffoliFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 3)),
                                 applyToffoliFunctor<Precision, true>(
                                     *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applyGeneratorPhaseShiftFunctor<Precision, false>(
                    *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applyGeneratorPhaseShiftFunctor<Precision, true>(
                    *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (inverse == false) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorIsingXXFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorIsingXXFunctor<Precision, true>(
                                     *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (inverse == false) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorIsingXYFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorIsingXYFunctor<Precision, true>(
                                     *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorIsingYYFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorIsingYYFunctor<Precision, true>(
                                     *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (inverse == false) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorIsingZZFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorIsingZZFunctor<Precision, true>(
                                     *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (inverse == false) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyGeneratorSingleExcitationFunctor<Precision, false>(
                    *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyGeneratorSingleExcitationFunctor<Precision, true>(
                    *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (inverse == false) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyGeneratorSingleExcitationMinusFunctor<Precision, false>(
                    *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyGeneratorSingleExcitationMinusFunctor<Precision, true>(
                    *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (inverse == false) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyGeneratorSingleExcitationPlusFunctor<Precision, false>(
                    *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyGeneratorSingleExcitationPlusFunctor<Precision, true>(
                    *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 4);

        if (inverse == false) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 4)),
                applyGeneratorDoubleExcitationFunctor<Precision, false>(
                    *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 4)),
                applyGeneratorDoubleExcitationFunctor<Precision, true>(
                    *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 4);

        if (inverse == false) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 4)),
                applyGeneratorDoubleExcitationMinusFunctor<Precision, false>(
                    *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 4)),
                applyGeneratorDoubleExcitationMinusFunctor<Precision, true>(
                    *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 4);

        if (inverse == false) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 4)),
                applyGeneratorDoubleExcitationPlusFunctor<Precision, false>(
                    *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 4)),
                applyGeneratorDoubleExcitationPlusFunctor<Precision, true>(
                    *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyPauliXFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applyPauliXFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyPauliYFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applyPauliYFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 1);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 1)),
                                 applyPauliZFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 1)),
                applyPauliZFunctor<Precision, true>(*data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyGeneratorControlledPhaseShiftFunctor<Precision, false>(
                    *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Util::exp2(num_qubits - 2)),
                applyGeneratorControlledPhaseShiftFunctor<Precision, true>(
                    *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (!inverse) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorCRXFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorCRXFunctor<Precision, true>(
                                     *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (inverse == false) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorCRYFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorCRYFunctor<Precision, true>(
                                     *data_, num_qubits, wires));
        }
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
        auto &&num_qubits = getNumQubits();
        assert(wires.size() == 2);

        if (inverse == false) {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorCRZFunctor<Precision, false>(
                                     *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(
                                     0, Util::exp2(num_qubits - 2)),
                                 applyGeneratorCRZFunctor<Precision, true>(
                                     *data_, num_qubits, wires));
        }
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
                Kokkos::RangePolicy<KokkosExecSpace>(0, Util::exp2(num_qubits)),
                applyGeneratorMultiRZFunctor<Precision, false>(
                    *data_, num_qubits, wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, Util::exp2(num_qubits)),
                applyGeneratorMultiRZFunctor<Precision, true>(
                    *data_, num_qubits, wires));
        }
        return -static_cast<Precision>(0.5);
    }

    /**
     * @brief Get the number of qubits of the state vector
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
     * @brief Get the Kokkos data of the state vector
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
     * @brief Copy data from the host space to the device space
     *
     */
    inline void HostToDevice(Kokkos::complex<Precision> *sv, size_t length) {
        Kokkos::deep_copy(*data_, UnmanagedComplexHostView(sv, length));
    }

    /**
     * @brief Copy data from the device space to the host space
     *
     */
    inline void DeviceToHost(Kokkos::complex<Precision> *sv, size_t length) {
        Kokkos::deep_copy(UnmanagedComplexHostView(sv, length), *data_);
    }

    /**
     * @brief Copy data from the device space to the device space
     *
     */
    inline void DeviceToDevice(KokkosVector vector_to_copy) {
        Kokkos::deep_copy(*data_, vector_to_copy);
    }

    static const size_t &GetCounts() { return counts_; }

  private:
    using GateFunc = std::function<void(const std::vector<size_t> &, bool,
                                        const std::vector<Precision> &)>;
    using GateMap = std::unordered_map<std::string, GateFunc>;
    const GateMap gates_;

    using GeneratorFunc = std::function<Precision(
        const std::vector<size_t> &, bool, const std::vector<Precision> &)>;
    using GeneratorMap = std::unordered_map<std::string, GeneratorFunc>;
    const GeneratorMap generator_;

    using ExpValFunc = std::function<Precision(const std::vector<size_t> &,
                                               const std::vector<Precision> &)>;
    using ExpValMap = std::unordered_map<std::string, ExpValFunc>;
    ExpValMap expval_funcs_;

    size_t num_qubits_;
    size_t length_;
    std::mutex counts_mutex_;
    inline static size_t counts_ = 0;
    std::unique_ptr<KokkosVector> data_;
};

}; // namespace Pennylane
