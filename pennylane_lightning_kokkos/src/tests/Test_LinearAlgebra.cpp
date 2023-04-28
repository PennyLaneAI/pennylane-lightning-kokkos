#include <complex>
#include <cstdio>
#include <vector>

#include "LinearAlgebraKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("Linear Algebra::SparseMV", "[Linear Algebra]", float,
                   double) {
    using cp_t = Kokkos::complex<TestType>;

    std::size_t num_qubits = 3;
    std::size_t data_size = Lightning_Kokkos::Util::exp2(num_qubits);

    std::vector<cp_t> vectors = {{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                 {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                 {0.3, 0.4}, {0.4, 0.5}};

    const std::vector<cp_t> result_refs = {
        {0.2, -0.1}, {-0.1, 0.2}, {0.2, 0.1}, {0.1, 0.2},
        {0.7, -0.2}, {-0.1, 0.6}, {0.6, 0.1}, {0.2, 0.7}};

    std::vector<size_t> indptr = {0, 2, 4, 6, 8, 10, 12, 14, 16};
    std::vector<size_t> indices = {0, 3, 1, 2, 1, 2, 0, 3,
                                   4, 7, 5, 6, 5, 6, 4, 7};
    std::vector<std::complex<TestType>> values = {
        {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
        {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0},
        {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
        {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0}};

    StateVectorKokkos<TestType> kokkos_vx{num_qubits};
    StateVectorKokkos<TestType> kokkos_vy{num_qubits};

    kokkos_vx.HostToDevice(vectors.data(), vectors.size());

    SECTION("Testing sparse matrix vector product:") {
        std::vector<cp_t> result(data_size);
        Lightning_Kokkos::Util::SparseMV_Kokkos<TestType>(
            kokkos_vx.getData(), kokkos_vy.getData(), values, indices, indptr);
        kokkos_vy.DeviceToHost(result.data(), result.size());

        for (std::size_t j = 0; j < Lightning_Kokkos::Util::exp2(num_qubits);
             j++) {
            CHECK(imag(result[j]) == Approx(imag(result_refs[j])));
            CHECK(real(result[j]) == Approx(real(result_refs[j])));
        }
    }
}

TEMPLATE_TEST_CASE("Linear Algebra::axpy_Kokkos", "[Linear Algebra]", float,
                   double) {
    using cp_t = Kokkos::complex<TestType>;

    std::size_t num_qubits = 3;

    std::size_t data_size = Lightning_Kokkos::Util::exp2(num_qubits);

    cp_t alpha = {2.0, 0.5};

    std::vector<cp_t> v0 = {{0.0, 0.0}, {0.1, -0.1}, {0.1, 0.1}, {0.2, 0.1},
                            {0.2, 0.2}, {0.3, 0.3},  {0.4, 0.3}, {0.5, 0.4}};

    std::vector<cp_t> v1 = {{-0.1, 0.2}, {0.2, -0.1}, {0.1, 0.2}, {0.2, 0.1},
                            {-0.2, 0.7}, {0.6, -0.1}, {0.1, 0.6}, {0.7, 0.2}};

    std::vector<cp_t> result_refs = {{-0.1, 0.2}, {0.45, -0.25}, {0.25, 0.45},
                                     {0.55, 0.4}, {0.1, 1.2},    {1.05, 0.65},
                                     {0.75, 1.4}, {1.5, 1.25}};

    StateVectorKokkos<TestType> kokkos_v0{num_qubits};
    StateVectorKokkos<TestType> kokkos_v1{num_qubits};

    kokkos_v0.HostToDevice(v0.data(), v0.size());
    kokkos_v1.HostToDevice(v1.data(), v1.size());

    SECTION("Testing imag of complex inner product:") {
        Lightning_Kokkos::Util::axpy_Kokkos<TestType>(
            alpha, kokkos_v0.getData(), kokkos_v1.getData(), v0.size());
        std::vector<cp_t> result(data_size);
        kokkos_v1.DeviceToHost(result.data(), result.size());

        for (std::size_t j = 0; j < Lightning_Kokkos::Util::exp2(num_qubits);
             j++) {
            CHECK(imag(result[j]) == Approx(imag(result_refs[j])));
            CHECK(real(result[j]) == Approx(real(result_refs[j])));
        }
    }
}
