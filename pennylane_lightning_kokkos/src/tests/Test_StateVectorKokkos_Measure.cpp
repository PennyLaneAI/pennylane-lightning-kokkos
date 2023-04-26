#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "MeasuresFunctors.hpp"
#include "MeasuresKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"

using namespace Pennylane;
namespace {} // namespace

TEMPLATE_TEST_CASE("Probabilities", "[Measures]", float, double) {
    // Probabilities calculated with Pennylane default_qbit:
    std::vector<std::pair<std::vector<size_t>, std::vector<TestType>>> input = {
        {{0, 1, 2},
         {0.67078706, 0.03062806, 0.0870997, 0.00397696, 0.17564072, 0.00801973,
          0.02280642, 0.00104134}},
        {{0, 2, 1},
         {0.67078706, 0.0870997, 0.03062806, 0.00397696, 0.17564072, 0.02280642,
          0.00801973, 0.00104134}},
        {{1, 0, 2},
         {0.67078706, 0.03062806, 0.17564072, 0.00801973, 0.0870997, 0.00397696,
          0.02280642, 0.00104134}},
        {{1, 2, 0},
         {0.67078706, 0.0870997, 0.17564072, 0.02280642, 0.03062806, 0.00397696,
          0.00801973, 0.00104134}},
        {{2, 0, 1},
         {0.67078706, 0.17564072, 0.03062806, 0.00801973, 0.0870997, 0.02280642,
          0.00397696, 0.00104134}},
        {{2, 1, 0},
         {0.67078706, 0.17564072, 0.0870997, 0.02280642, 0.03062806, 0.00801973,
          0.00397696, 0.00104134}},
        {{0, 1}, {0.70141512, 0.09107666, 0.18366045, 0.02384776}},
        {{0, 2}, {0.75788676, 0.03460502, 0.19844714, 0.00906107}},
        {{1, 2}, {0.84642778, 0.0386478, 0.10990612, 0.0050183}},
        {{2, 1}, {0.84642778, 0.10990612, 0.0386478, 0.0050183}},
        {{0}, {0.79249179, 0.20750821}},
        {{1}, {0.88507558, 0.11492442}},
        {{2}, {0.9563339, 0.0436661}}};

    // Defining the State Vector that will be measured.
    const std::size_t num_qubits = 3;
    auto measure_sv = Initializing_StateVector<TestType>(num_qubits);

    SECTION("Looping over different wire configurations:") {
        auto m =
            Lightning_Kokkos::Simulators::MeasuresKokkos<TestType>(measure_sv);
        for (const auto &term : input) {
            auto probabilities = m.probs(term.first);
            REQUIRE_THAT(term.second,
                         Catch::Approx(probabilities).margin(1e-6));
        }
    }
}

TEST_CASE("Test tensor transposition", "[Measure]") {
    // Transposition axes and expected result.
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> input = {
        {{0, 1, 2}, {0, 1, 2, 3, 4, 5, 6, 7}},
        {{0, 2, 1}, {0, 2, 1, 3, 4, 6, 5, 7}},
        {{1, 0, 2}, {0, 1, 4, 5, 2, 3, 6, 7}},
        {{1, 2, 0}, {0, 4, 1, 5, 2, 6, 3, 7}},
        {{2, 0, 1}, {0, 2, 4, 6, 1, 3, 5, 7}},
        {{2, 1, 0}, {0, 4, 2, 6, 1, 5, 3, 7}},
        {{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        {{0, 1, 3, 2}, {0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15}},
        {{0, 2, 1, 3}, {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}},
        {{0, 2, 3, 1}, {0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15}},
        {{0, 3, 1, 2}, {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15}},
        {{0, 3, 2, 1}, {0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15}},
        {{1, 0, 2, 3}, {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15}},
        {{1, 0, 3, 2}, {0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15}},
        {{1, 2, 0, 3}, {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}},
        {{1, 2, 3, 0}, {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}},
        {{1, 3, 0, 2}, {0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15}},
        {{1, 3, 2, 0}, {0, 8, 2, 10, 1, 9, 3, 11, 4, 12, 6, 14, 5, 13, 7, 15}},
        {{2, 0, 1, 3}, {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15}},
        {{2, 0, 3, 1}, {0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15}},
        {{2, 1, 0, 3}, {0, 1, 8, 9, 4, 5, 12, 13, 2, 3, 10, 11, 6, 7, 14, 15}},
        {{2, 1, 3, 0}, {0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15}},
        {{2, 3, 0, 1}, {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}},
        {{2, 3, 1, 0}, {0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15}},
        {{3, 0, 1, 2}, {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}},
        {{3, 0, 2, 1}, {0, 4, 2, 6, 8, 12, 10, 14, 1, 5, 3, 7, 9, 13, 11, 15}},
        {{3, 1, 0, 2}, {0, 2, 8, 10, 4, 6, 12, 14, 1, 3, 9, 11, 5, 7, 13, 15}},
        {{3, 1, 2, 0}, {0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15}},
        {{3, 2, 0, 1}, {0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15}},
        {{3, 2, 1, 0}, {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}}};

    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    using UnmanagedSizeTHostView =
        Kokkos::View<size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    SECTION("Looping over different wire configurations:") {
        for (auto &term : input) {
            // Defining a tensor to be transposed.
            std::vector<size_t> indices(1U << term.first.size());
            std::iota(indices.begin(), indices.end(), 0);

            std::vector<size_t> results(indices.size());

            Kokkos::View<size_t *> d_indices("d_indices", indices.size());
            Kokkos::View<size_t *> d_results("d_results", indices.size());
            Kokkos::View<size_t *> d_wires("d_wires", term.first.size());
            Kokkos::View<size_t *> d_trans_index("d_trans_index",
                                                 indices.size());

            Kokkos::deep_copy(d_indices, UnmanagedSizeTHostView(
                                             indices.data(), indices.size()));
            Kokkos::deep_copy(
                d_wires,
                UnmanagedSizeTHostView(term.first.data(), term.first.size()));

            using MDPolicyType_2D =
                Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>;

            MDPolicyType_2D mdpolicy_2d1(
                {{0, 0}}, {{static_cast<int>(indices.size()),
                            static_cast<int>(term.first.size())}});

            const int num_wires = term.first.size();

            Kokkos::parallel_for(
                "TransIndex", mdpolicy_2d1,
                getTransposedIndexFunctor(d_wires, d_trans_index, num_wires));

            Kokkos::parallel_for(
                "Transpose",
                Kokkos::RangePolicy<KokkosExecSpace>(0, indices.size()),
                getTransposedFunctor<size_t>(d_results, d_indices,
                                             d_trans_index));

            Kokkos::deep_copy(
                UnmanagedSizeTHostView(results.data(), results.size()),
                d_results);
            REQUIRE(term.second == results);
        }
    }
}
