#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "MeasuresKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"

using namespace Pennylane;
using namespace Pennylane::Lightning_Kokkos::Simulators;

TEMPLATE_TEST_CASE("Test variance of NamedObs", "[StateVectorKokkos_Var]",
                   float, double) {
    const std::size_t num_qubits = 2;
    SECTION("var(PauliX[0])") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = MeasuresKokkos<TestType>(kokkos_sv);

        kokkos_sv.applyOperation("RX", {0}, false, {0.7});
        kokkos_sv.applyOperation("RY", {0}, false, {0.7});
        kokkos_sv.applyOperation("RX", {1}, false, {0.5});
        kokkos_sv.applyOperation("RY", {1}, false, {0.5});

        auto ob = NamedObsKokkos<TestType>("PauliX", {0});
        auto res = m.var(ob);
        auto expected = TestType(0.7572222074);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliY[0])") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = MeasuresKokkos<TestType>(kokkos_sv);

        kokkos_sv.applyOperation("RX", {0}, false, {0.7});
        kokkos_sv.applyOperation("RY", {0}, false, {0.7});
        kokkos_sv.applyOperation("RX", {1}, false, {0.5});
        kokkos_sv.applyOperation("RY", {1}, false, {0.5});

        auto ob = NamedObsKokkos<TestType>("PauliY", {0});
        auto res = m.var(ob);
        auto expected = TestType(0.5849835715);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliZ[1])") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = MeasuresKokkos<TestType>(kokkos_sv);

        kokkos_sv.applyOperation("RX", {0}, false, {0.7});
        kokkos_sv.applyOperation("RY", {0}, false, {0.7});
        kokkos_sv.applyOperation("RX", {1}, false, {0.5});
        kokkos_sv.applyOperation("RY", {1}, false, {0.5});

        auto ob = NamedObsKokkos<TestType>("PauliZ", {1});
        auto res = m.var(ob);
        auto expected = TestType(0.4068672016);
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_TEST_CASE("Test variance of HermitianObs", "[StateVectorKokkos_Var]",
                   float, double) {
    const std::size_t num_qubits = 3;
    SECTION("Using var") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = MeasuresKokkos<TestType>(kokkos_sv);

        kokkos_sv.applyOperation("RX", {0}, false, {0.7});
        kokkos_sv.applyOperation("RY", {0}, false, {0.7});
        kokkos_sv.applyOperation("RX", {1}, false, {0.5});
        kokkos_sv.applyOperation("RY", {1}, false, {0.5});
        kokkos_sv.applyOperation("RX", {2}, false, {0.3});
        kokkos_sv.applyOperation("RY", {2}, false, {0.3});

        const TestType theta = M_PI / 2;
        const TestType c = std::cos(theta / 2);
        const TestType js = std::sin(-theta / 2);
        std::vector<std::complex<TestType>> matrix(16);
        matrix[0] = c;
        matrix[1] = {0, js};
        matrix[4] = {0, js};
        matrix[5] = c;
        matrix[10] = {1, 0};
        matrix[15] = {1, 0};

        auto ob = HermitianObsKokkos<TestType>(matrix, {0, 2});
        auto res = m.var(ob);
        auto expected = TestType(0.4103533486);
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_TEST_CASE("Test variance of TensorProdObs", "[StateVectorKokkos_Var]",
                   float, double) {
    const std::size_t num_qubits = 3;
    SECTION("Using var") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = MeasuresKokkos<TestType>(kokkos_sv);

        kokkos_sv.applyOperation("RX", {0}, false, {0.5});
        kokkos_sv.applyOperation("RY", {0}, false, {0.5});
        kokkos_sv.applyOperation("RX", {1}, false, {0.2});
        kokkos_sv.applyOperation("RY", {1}, false, {0.2});

        auto X0 = std::make_shared<NamedObsKokkos<TestType>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObsKokkos<TestType>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = TensorProdObsKokkos<TestType>::create({X0, Z1});
        auto res = m.var(*ob);
        auto expected = TestType(0.836679);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_TEST_CASE("Test variance of HamiltonianObs", "[StateVectorKokkos_Var]",
                   float, double) {
    SECTION("Using var") {
        std::vector<Kokkos::complex<TestType>> init_state{
            {0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.1, 0.2},
            {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        auto m = MeasuresKokkos<TestType>(kokkos_sv);

        auto X0 = std::make_shared<NamedObsKokkos<TestType>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObsKokkos<TestType>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = HamiltonianKokkos<TestType>::create({0.3, 0.5}, {X0, Z1});
        auto res = m.var(*ob);
        auto expected = TestType(0.224604);
        CHECK(expected == Approx(res));
    }
}
