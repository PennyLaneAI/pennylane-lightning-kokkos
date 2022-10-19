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

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueIdentity",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    const std::size_t num_qubits = 3;
    auto ONE = TestType(1);
    StateVectorKokkos<TestType> kokkos_sv{num_qubits};

    SECTION("Apply directly") {
        kokkos_sv.applyOperation("Hadamard", {0}, false);
        kokkos_sv.applyOperation("CNOT", {0, 1}, false);
        kokkos_sv.applyOperation("CNOT", {1, 2}, false);
        auto res = kokkos_sv.getExpectationValueIdentity({0});
        CHECK(res == Approx(ONE));
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValuePauliX",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const std::size_t num_qubits = 3;

        auto ZERO = TestType(0);
        auto ONE = TestType(1);

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("CNOT", {0, 1}, false);
            kokkos_sv.applyOperation("CNOT", {1, 2}, false);
            auto res = kokkos_sv.getExpectationValuePauliX({0});
            CHECK(res == ZERO);
        }
        SECTION("Apply directly: Plus states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("Hadamard", {1}, false);
            kokkos_sv.applyOperation("Hadamard", {2}, false);
            auto res = kokkos_sv.getExpectationValuePauliX({0});
            CHECK(res == Approx(ONE));
        }
        SECTION("Apply directly: Minus states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyPauliX({0}, false);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyPauliX({1}, false);
            kokkos_sv.applyOperation("Hadamard", {1}, false);
            kokkos_sv.applyPauliX({2}, false);
            kokkos_sv.applyOperation("Hadamard", {2}, false);
            auto res = kokkos_sv.getExpectationValuePauliX({0});
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValuePauliY",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const std::size_t num_qubits = 3;

        auto ZERO = TestType(0);
        auto ONE = TestType(1);
        auto PI = TestType(M_PI);

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("CNOT", {0, 1}, false);
            kokkos_sv.applyOperation("CNOT", {1, 2}, false);
            auto res = kokkos_sv.getExpectationValuePauliY({0});
            CHECK(res == ZERO);
        }
        SECTION("Apply directly: Plus i states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyRX({0}, false, {-PI / 2});
            kokkos_sv.applyRX({1}, false, {-PI / 2});
            kokkos_sv.applyRX({2}, false, {-PI / 2});
            auto res = kokkos_sv.getExpectationValuePauliY({0});
            CHECK(res == Approx(ONE));
        }
        SECTION("Apply directly: Minus i states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyRX({0}, false, {PI / 2});
            kokkos_sv.applyRX({1}, false, {PI / 2});
            kokkos_sv.applyRX({2}, false, {PI / 2});
            auto res = kokkos_sv.getExpectationValuePauliY({0});
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValuePauliZ",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const std::size_t num_qubits = 3;

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("CNOT", {0, 1}, false);
            kokkos_sv.applyOperation("CNOT", {1, 2}, false);
            auto res = kokkos_sv.getExpectationValuePauliZ({0});
            CHECK(res == 0); // A 0-result is not a good test.
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueHadamard",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const std::size_t num_qubits = 3;
        auto INVSQRT2 = TestType(0.707106781186547524401);

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto res = kokkos_sv.getExpectationValueHadamard({0});
            CHECK(res == INVSQRT2);
        }
        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyPauliX({0});
            auto res = kokkos_sv.getExpectationValueHadamard({0});
            CHECK(res == -INVSQRT2);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueSingleQubitOp",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const std::size_t num_qubits = 3;

        auto INVSQRT2 = TestType(0.707106781186547524401);

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            Kokkos::View<Kokkos::complex<TestType> *> opMatDevice("opMat", 4);
            Kokkos::View<Kokkos::complex<TestType> *, Kokkos::HostSpace> opMat(
                "opMatHost", 4);

            const TestType theta = M_PI / 2;
            const TestType c = std::cos(theta / 2);
            const TestType js = std::sin(-theta / 2);

            opMat[0] = c;
            opMat[1] = Kokkos::complex(static_cast<TestType>(0), js);
            opMat[2] = Kokkos::complex(static_cast<TestType>(0), js);
            opMat[3] = c;

            Kokkos::deep_copy(opMatDevice, opMat);
            auto res =
                kokkos_sv.getExpectationValueSingleQubitOp(opMatDevice, {0});
            CHECK(res == INVSQRT2);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueTwoQubitOp",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const std::size_t num_qubits = 3;
        auto INVSQRT2 = TestType(0.707106781186547524401);

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};

            Kokkos::View<Kokkos::complex<TestType> *> opMatDevice("opMat", 16);
            Kokkos::View<Kokkos::complex<TestType> *, Kokkos::HostSpace> opMat(
                "opMatHost", 16);

            const TestType theta = M_PI / 2;
            const TestType c = std::cos(theta / 2);
            const TestType js = std::sin(-theta / 2);

            opMat[0] = c;
            opMat[1] = Kokkos::complex<TestType>(static_cast<TestType>(0), js);
            opMat[4] = Kokkos::complex<TestType>(static_cast<TestType>(0), js);
            opMat[5] = c;
            opMat[10] = Kokkos::complex<TestType>(static_cast<TestType>(1), 0);
            opMat[15] = Kokkos::complex<TestType>(static_cast<TestType>(1), 0);

            Kokkos::deep_copy(opMatDevice, opMat);
            auto res =
                kokkos_sv.getExpectationValueTwoQubitOp(opMatDevice, {0, 1});
            CHECK(res == INVSQRT2);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::Hamiltonian_expval",
                   "[StateVectorKokkos_Nonparam]", double) {
    using cp_t = Kokkos::complex<TestType>;
    const std::size_t num_qubits = 3;
    SECTION("GetExpectionIdentity") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        std::vector<size_t> wires{0, 1, 2};

        kokkos_sv.applyHadamard({0}, false);
        kokkos_sv.applyCNOT({0, 1}, false);
        kokkos_sv.applyCNOT({1, 2}, false);

        size_t matrix_dim = static_cast<size_t>(1U) << num_qubits;
        std::vector<cp_t> matrix(matrix_dim * matrix_dim);

        for (size_t i = 0; i < matrix.size(); i++) {
            if (i % matrix_dim == i / matrix_dim)
                matrix[i] = std::complex<TestType>(1, 0);
            else
                matrix[i] = std::complex<TestType>(0, 0);
        }

        auto results = kokkos_sv.getExpectationValue(wires, matrix);
        cp_t expected(1, 0);
        CHECK(real(expected) == Approx(results).epsilon(1e-7));
    }

    SECTION("GetExpectionHermitianMatrix") {
        using cp_t = Kokkos::complex<TestType>;
        std::vector<cp_t> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                     {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                     {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        std::vector<size_t> wires{0, 1, 2};
        std::vector<cp_t> matrix{
            {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0}};

        auto results = kokkos_sv.getExpectationValue(wires, matrix);
        cp_t expected(1.263000, -1.011000);

        CHECK(real(expected) == Approx(results).epsilon(1e-7));
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::Hamiltonian_expval_Sparse",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    using cp_t = Kokkos::complex<TestType>;
    SECTION("GetExpectionSparse") {
        std::vector<cp_t> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                     {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                     {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};

        std::vector<size_t> index_ptr = {0, 2, 4, 6, 8, 10, 12, 14, 16};
        std::vector<size_t> indices = {0, 3, 1, 2, 1, 2, 0, 3,
                                       4, 7, 5, 6, 5, 6, 4, 7};
        std::vector<Kokkos::complex<TestType>> values = {
            {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
            {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0},
            {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
            {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0}};

        auto result = kokkos_sv.getExpectationValue(values, indices, index_ptr);

        TestType expected = 1;

        CHECK(expected == Approx(result).epsilon(1e-7));
    }
}
