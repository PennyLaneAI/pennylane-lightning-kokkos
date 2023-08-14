#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "ExpValFunctors.hpp"
#include "LinearAlgebraKokkos.hpp"
#include "MeasuresFunctors.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"

namespace Pennylane::Lightning_Kokkos::Simulators {

template <class Precision> class MeasuresKokkos {

  private:
    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    using KokkosVector = Kokkos::View<Kokkos::complex<Precision> *>;
    using KokkosSizeTVector = Kokkos::View<size_t *>;
    using UnmanagedSizeTHostView =
        Kokkos::View<size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedPrecisionHostView =
        Kokkos::View<Precision *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstComplexHostView =
        Kokkos::View<const Kokkos::complex<Precision> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstSizeTHostView =
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ExpValFunc = std::function<Precision(const std::vector<size_t> &,
                                               const std::vector<Precision> &)>;
    using ExpValMap = std::unordered_map<std::string, ExpValFunc>;

    const StateVectorKokkos<Precision> &original_sv;
    ExpValMap expval_funcs;

  public:
    explicit MeasuresKokkos(const StateVectorKokkos<Precision> &state_vector)
        : original_sv{state_vector},
          expval_funcs{{"Identity",
                        [&](auto &&wires, auto &&params) {
                            return getExpectationValueIdentity(
                                std::forward<decltype(wires)>(wires),
                                std::forward<decltype(params)>(params));
                        }},
                       {"PauliX",
                        [&](auto &&wires, auto &&params) {
                            return getExpectationValuePauliX(
                                std::forward<decltype(wires)>(wires),
                                std::forward<decltype(params)>(params));
                        }},
                       {"PauliY",
                        [&](auto &&wires, auto &&params) {
                            return getExpectationValuePauliY(
                                std::forward<decltype(wires)>(wires),
                                std::forward<decltype(params)>(params));
                        }},
                       {"PauliZ",
                        [&](auto &&wires, auto &&params) {
                            return getExpectationValuePauliZ(
                                std::forward<decltype(wires)>(wires),
                                std::forward<decltype(params)>(params));
                        }},
                       {"Hadamard", [&](auto &&wires, auto &&params) {
                            return getExpectationValueHadamard(
                                std::forward<decltype(wires)>(wires),
                                std::forward<decltype(params)>(params));
                        }}} {};

    /**
     * @brief Calculate the expectation value of a named observable.
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

        if (expval_funcs.find(obsName) != expval_funcs.end()) {
            return expval_funcs.at(obsName)(local_wires, par);
        }

        KokkosVector matrix("gate_matrix", gate_matrix.size());
        Kokkos::deep_copy(matrix, UnmanagedConstComplexHostView(
                                      gate_matrix.data(), gate_matrix.size()));
        return getExpectationValueMultiQubitOp(matrix, wires, par);
    }

    /**
     * @brief Calculate the expectation value of a matrix. Typically,
     * this function will be used for dense Hamiltonians.
     *
     * @param obsName observable name
     * @param wires wires the observable acts on
     * @param params parameters for the observable
     * @param gate_matrix optional matrix
     */
    auto getExpectationValue(
        const std::vector<size_t> &wires,
        const std::vector<Kokkos::complex<Precision>> &gate_matrix) {

        auto &&par = std::vector<Precision>{0.0};
        KokkosVector matrix("gate_matrix", gate_matrix.size());
        Kokkos::deep_copy(matrix, UnmanagedConstComplexHostView(
                                      gate_matrix.data(), gate_matrix.size()));
        return getExpectationValueMultiQubitOp(matrix, wires, par);
    }

    /**
     * @brief Calculate the expectation value of a sparse Hamiltonian in CSR
     * format. Typically, this function will be used for dense hamiltonians.
     *
     * @param obsName observable name
     * @param wires wires the observable acts on
     * @param params parameters for the observable
     * @param gate_matrix optional matrix
     */
    auto
    getExpectationValue(const std::vector<Kokkos::complex<Precision>> &data,
                        const std::vector<size_t> &indices,
                        const std::vector<size_t> &index_ptr) {
        const Kokkos::View<Kokkos::complex<Precision> *> arr_data =
            original_sv.getData();
        Precision expval = 0;
        KokkosSizeTVector kok_indices("indices", indices.size());
        KokkosSizeTVector kok_index_ptr("index_ptr", index_ptr.size());
        KokkosVector kok_data("data", data.size());

        Kokkos::deep_copy(
            kok_data, UnmanagedConstComplexHostView(data.data(), data.size()));
        Kokkos::deep_copy(kok_indices, UnmanagedConstSizeTHostView(
                                           indices.data(), indices.size()));
        Kokkos::deep_copy(
            kok_index_ptr,
            UnmanagedConstSizeTHostView(index_ptr.data(), index_ptr.size()));

        Kokkos::parallel_reduce(
            index_ptr.size() - 1,
            getExpectationValueSparseFunctor<Precision>(
                arr_data, kok_data, kok_indices, kok_index_ptr),
            expval);
        return expval;
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
        const size_t num_qubits = original_sv.getNumQubits();
        const Kokkos::View<Kokkos::complex<Precision> *> arr_data =
            original_sv.getData();
        Precision expval = 0;
        Kokkos::parallel_reduce(
            Lightning_Kokkos::Util::exp2(num_qubits),
            getExpectationValueIdentityFunctor(arr_data, num_qubits, wires),
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
        const size_t num_qubits = original_sv.getNumQubits();
        const Kokkos::View<Kokkos::complex<Precision> *> arr_data =
            original_sv.getData();
        Precision expval = 0;
        Kokkos::parallel_reduce(
            Lightning_Kokkos::Util::exp2(num_qubits - 1),
            getExpectationValuePauliXFunctor(arr_data, num_qubits, wires),
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
        const size_t num_qubits = original_sv.getNumQubits();
        const Kokkos::View<Kokkos::complex<Precision> *> arr_data =
            original_sv.getData();
        Precision expval = 0;
        Kokkos::parallel_reduce(
            Lightning_Kokkos::Util::exp2(num_qubits - 1),
            getExpectationValuePauliYFunctor(arr_data, num_qubits, wires),
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
        const size_t num_qubits = original_sv.getNumQubits();
        const Kokkos::View<Kokkos::complex<Precision> *> arr_data =
            original_sv.getData();
        Precision expval = 0;
        Kokkos::parallel_reduce(
            Lightning_Kokkos::Util::exp2(num_qubits - 1),
            getExpectationValuePauliZFunctor(arr_data, num_qubits, wires),
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
        const size_t num_qubits = original_sv.getNumQubits();
        const Kokkos::View<Kokkos::complex<Precision> *> arr_data =
            original_sv.getData();
        Precision expval = 0;
        Kokkos::parallel_reduce(
            Lightning_Kokkos::Util::exp2(num_qubits - 1),
            getExpectationValueHadamardFunctor(arr_data, num_qubits, wires),
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
        const size_t num_qubits = original_sv.getNumQubits();
        Kokkos::View<Kokkos::complex<Precision> *> arr_data =
            original_sv.getData();
        Precision expval = 0;
        Kokkos::parallel_reduce(
            Lightning_Kokkos::Util::exp2(num_qubits - 1),
            getExpectationValueSingleQubitOpFunctor<Precision>(
                arr_data, num_qubits, matrix, wires),
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
        const size_t num_qubits = original_sv.getNumQubits();
        Kokkos::View<Kokkos::complex<Precision> *> arr_data =
            original_sv.getData();
        Precision expval = 0;
        Kokkos::parallel_reduce(Lightning_Kokkos::Util::exp2(num_qubits - 2),
                                getExpectationValueTwoQubitOpFunctor<Precision>(
                                    arr_data, num_qubits, matrix, wires),
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
            Kokkos::View<const size_t *, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                wires_host(wires.data(), wires.size());
            const size_t num_qubits = original_sv.getNumQubits();
            const Kokkos::View<Kokkos::complex<Precision> *> arr_data =
                original_sv.getData();

            Kokkos::View<size_t *> wires_view("wires_view", wires.size());
            Kokkos::deep_copy(wires_view, wires_host);
            Precision expval = 0;
            Kokkos::parallel_reduce(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, Lightning_Kokkos::Util::exp2(num_qubits - wires.size())),
                getExpectationValueMultiQubitOpFunctor(arr_data, num_qubits,
                                                       matrix, wires_view),
                expval);
            return expval;
        }
    }

    /**
     * @brief Calculate expectation value for a general Observable.
     *
     * @param ob Observable.
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const ObservableKokkos<Precision> &ob) {
        StateVectorKokkos<Precision> ob_sv(original_sv.getNumQubits());
        ob_sv.DeviceToDevice(original_sv.getData());
        ob.applyInPlace(ob_sv);
        return Pennylane::Lightning_Kokkos::Util::getRealOfComplexInnerProduct(
            original_sv.getData(), ob_sv.getData());
    }

    /**
     * @brief Calculate variance of a general Observable.
     *
     * @param ob Observable.
     * @return Variance with respect to the given observable.
     */
    auto var(const ObservableKokkos<Precision> &ob) -> Precision {
        StateVectorKokkos<Precision> ob_sv(original_sv.getNumQubits());
        ob_sv.DeviceToDevice(original_sv.getData());
        ob.applyInPlace(ob_sv);

        const Precision mean_square =
            Pennylane::Lightning_Kokkos::Util::getRealOfComplexInnerProduct(
                ob_sv.getData(), ob_sv.getData());
        const Precision squared_mean = static_cast<Precision>(std::pow(
            Pennylane::Lightning_Kokkos::Util::getRealOfComplexInnerProduct(
                original_sv.getData(), ob_sv.getData()),
            2));
        return (mean_square - squared_mean);
    }

    /**
     * @brief Probabilities of each computational basis state.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    auto probs() -> std::vector<Precision> {
        const size_t N = original_sv.getLength();

        Kokkos::View<Kokkos::complex<Precision> *> arr_data =
            original_sv.getData();
        Kokkos::View<Precision *> d_probability("d_probability", N);

        // Compute probability distribution from StateVector using
        // Kokkos::parallel_for
        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(0, N),
            getProbFunctor<Precision>(arr_data, d_probability));

        std::vector<Precision> probabilities(N, 0);

        Kokkos::deep_copy(UnmanagedPrecisionHostView(probabilities.data(),
                                                     probabilities.size()),
                          d_probability);
        return probabilities;
    }

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     */
    auto probs(const std::vector<size_t> &wires) {
        using MDPolicyType_2D =
            Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>;

        //  Determining probabilities for the sorted wires.
        const Kokkos::View<Kokkos::complex<Precision> *> arr_data =
            original_sv.getData();
        const size_t num_qubits = original_sv.getNumQubits();

        std::vector<size_t> sorted_ind_wires(wires);
        const bool is_sorted_wires =
            std::is_sorted(sorted_ind_wires.begin(), sorted_ind_wires.end());
        std::vector<size_t> sorted_wires(wires);

        if (!is_sorted_wires) {
            sorted_ind_wires = Lightning_Kokkos::Util::sorting_indices(wires);
            for (size_t pos = 0; pos < wires.size(); pos++)
                sorted_wires[pos] = wires[sorted_ind_wires[pos]];
        }

        std::vector<size_t> all_indices =
            Lightning_Kokkos::Util::generateBitsPatterns(sorted_wires,
                                                         num_qubits);

        std::vector<size_t> all_offsets =
            Lightning_Kokkos::Util::generateBitsPatterns(
                Lightning_Kokkos::Util::getIndicesAfterExclusion(sorted_wires,
                                                                 num_qubits),
                num_qubits);

        Kokkos::View<Precision *> d_probabilities("d_probabilities",
                                                  all_indices.size());

        Kokkos::View<size_t *> d_sorted_ind_wires("d_sorted_ind_wires",
                                                  sorted_ind_wires.size());
        Kokkos::View<size_t *> d_all_indices("d_all_indices",
                                             all_indices.size());
        Kokkos::View<size_t *> d_all_offsets("d_all_offsets",
                                             all_offsets.size());

        Kokkos::deep_copy(
            d_all_indices,
            UnmanagedSizeTHostView(all_indices.data(), all_indices.size()));
        Kokkos::deep_copy(
            d_all_offsets,
            UnmanagedSizeTHostView(all_offsets.data(), all_offsets.size()));
        Kokkos::deep_copy(d_sorted_ind_wires,
                          UnmanagedSizeTHostView(sorted_ind_wires.data(),
                                                 sorted_ind_wires.size()));

        const int num_all_indices =
            all_indices.size(); // int is required by Kokkos::MDRangePolicy
        const int num_all_offsets = all_offsets.size();

        MDPolicyType_2D mdpolicy_2d0({{0, 0}},
                                     {{num_all_indices, num_all_offsets}});

        Kokkos::parallel_for(
            "Set_Prob", mdpolicy_2d0,
            getSubProbFunctor<Precision>(arr_data, d_probabilities,
                                         d_all_indices, d_all_offsets));

        std::vector<Precision> probabilities(all_indices.size(), 0);

        if (is_sorted_wires) {
            Kokkos::deep_copy(UnmanagedPrecisionHostView(probabilities.data(),
                                                         probabilities.size()),
                              d_probabilities);
            return probabilities;
        } else {
            Kokkos::View<Precision *> transposed_tensor("transposed_tensor",
                                                        all_indices.size());

            Kokkos::View<size_t *> d_trans_index("d_trans_index",
                                                 all_indices.size());

            const int num_trans_tensor = transposed_tensor.size();
            const int num_sorted_ind_wires = sorted_ind_wires.size();

            MDPolicyType_2D mdpolicy_2d1(
                {{0, 0}}, {{num_trans_tensor, num_sorted_ind_wires}});

            Kokkos::parallel_for(
                "TransIndex", mdpolicy_2d1,
                getTransposedIndexFunctor(d_sorted_ind_wires, d_trans_index,
                                          num_sorted_ind_wires));

            Kokkos::parallel_for(
                "Transpose",
                Kokkos::RangePolicy<KokkosExecSpace>(0, num_trans_tensor),
                getTransposedFunctor<Precision>(
                    transposed_tensor, d_probabilities, d_trans_index));

            Kokkos::deep_copy(UnmanagedPrecisionHostView(probabilities.data(),
                                                         probabilities.size()),
                              transposed_tensor);

            return probabilities;
        }
    }

    /**
     * @brief  Inverse transform sampling method for samples.
     * Reference https://en.wikipedia.org/wiki/Inverse_transform_sampling
     *
     * @param num_samples Number of Samples
     *
     * @return std::vector<size_t> to the samples.
     * Each sample has a length equal to the number of qubits. Each sample can
     * be accessed using the stride sample_id*num_qubits, where sample_id is a
     * number between 0 and num_samples-1.
     */
    auto generate_samples(size_t num_samples) -> std::vector<size_t> {

        const size_t num_qubits = original_sv.getNumQubits();
        const size_t N = original_sv.getLength();

        Kokkos::View<Kokkos::complex<Precision> *> arr_data =
            original_sv.getData();
        Kokkos::View<Precision *> probability("probability", N);
        Kokkos::View<size_t *> samples("num_samples", num_samples * num_qubits);

        // Compute probability distribution from StateVector using
        // Kokkos::parallel_for
        Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(0, N),
                             getProbFunctor<Precision>(arr_data, probability));

        // Convert probability distribution to cumulative distribution using
        // Kokkos:: parallel_scan
        Kokkos::parallel_scan(Kokkos::RangePolicy<KokkosExecSpace>(0, N),
                              getCDFFunctor<Precision>(probability));

        // Sampling using Random_XorShift64_Pool
        Kokkos::Random_XorShift64_Pool<> rand_pool(5374857);

        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(0, num_samples),
            Sampler<Precision, Kokkos::Random_XorShift64_Pool>(
                samples, probability, rand_pool, num_qubits, N));

        std::vector<size_t> samples_h(num_samples * num_qubits);

        using UnmanagedSize_tHostView =
            Kokkos::View<size_t *, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        Kokkos::deep_copy(
            UnmanagedSize_tHostView(samples_h.data(), samples_h.size()),
            samples);

        return samples_h;
    }
};

} // namespace Pennylane::Lightning_Kokkos::Simulators
