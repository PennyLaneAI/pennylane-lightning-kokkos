#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"

using namespace Pennylane;
namespace {} // namespace

TEMPLATE_TEST_CASE("Probabilities", "[Measures]", float, double) {
    // Probabilities calculated with Pennylane default_qbit:
    std::vector<std::vector<TestType>> expected_probabilities = {
        {0.67078706, 0.03062806, 0.0870997, 0.00397696, 0.17564072, 0.00801973,
         0.02280642, 0.00104134}, // probs(0,1,2)
        {0.67078706, 0.0870997, 0.03062806, 0.00397696, 0.17564072, 0.02280642,
         0.00801973, 0.00104134}, // probs(0,2,1)
        {0.67078706, 0.03062806, 0.17564072, 0.00801973, 0.0870997, 0.00397696,
         0.02280642, 0.00104134}, // probs(1,0,2)
        {0.67078706, 0.17564072, 0.0870997, 0.02280642, 0.03062806, 0.00801973,
         0.00397696, 0.00104134}, // probs(2,1,0)
        {0.67078706, 0.17564072, 0.03062806, 0.00801973, 0.0870997, 0.02280642,
         0.00397696, 0.00104134}, // probs(2,0,1)
        {0.67078706, 0.0870997, 0.17564072, 0.02280642, 0.03062806, 0.00397696,
         0.00801973, 0.00104134},                         // probs(1,2,0)
        {0.70141512, 0.09107666, 0.18366045, 0.02384776}, // probs(0,1)
        {0.75788676, 0.03460502, 0.19844714, 0.00906107}, // probs(0,2)
        {0.84642778, 0.0386478, 0.10990612, 0.0050183},   // probs(1,2)
        {0.84642778, 0.10990612, 0.0386478, 0.0050183},   // probs(2,1)
        {0.79249179, 0.20750821},                         // probs(0)
        {0.88507558, 0.11492442},                         // probs(1)
        {0.9563339, 0.0436661}                            // probs(2)
    };

    std::vector<std::vector<size_t>> wires_set = {
        {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {2, 1, 0}, {2, 0, 1},
        {1, 2, 0}, {0, 1},    {0, 2},    {1, 2},    {2, 1},
        {0},       {1},       {2}};

    // Defining the State Vector that will be measured.
    const std::size_t num_qubits = 3;
    StateVectorKokkos<TestType> measure_sv{num_qubits};

    std::vector<std::string> gates;
    std::vector<std::vector<size_t>> wires;
    std::vector<bool> inv_op(num_qubits * 2, false);
    std::vector<std::vector<TestType>> phase;

    TestType initial_phase = 0.7;
    for (size_t n_qubit = 0; n_qubit < num_qubits; n_qubit++) {
        gates.emplace_back("RX");
        gates.emplace_back("RY");

        wires.push_back({n_qubit});
        wires.push_back({n_qubit});

        phase.push_back({initial_phase});
        phase.push_back({initial_phase});
        initial_phase -= 0.2;
    }

    measure_sv.applyOperation(gates, wires, inv_op, phase);

    SECTION("Looping over different wire configurations:") {
        for (size_t i = 0; i < expected_probabilities.size(); i++) {
            const bool sorted_or_not =
                std::is_sorted(wires_set[i].begin(), wires_set[i].end());
            auto probabilities = measure_sv.probs(wires_set[i], sorted_or_not);
            REQUIRE_THAT(probabilities,
                         Catch::Approx(expected_probabilities[i]).margin(1e-6));
        }
    }
}
