#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "GatesHost.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"

using namespace Pennylane;
using namespace Pennylane::Gates;
namespace {} // namespace

TEMPLATE_TEST_CASE("StateVectorKokkos::CopyConstructor",
                   "[StateVectorKokkos_Nonparam]", float, double) {

    {
        const std::size_t num_qubits = 3;
        StateVectorKokkos<TestType> kokkos_sv_1{num_qubits};
        kokkos_sv_1.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                   {{0}, {1}, {2}},
                                   {{false}, {false}, {false}});
        StateVectorKokkos<TestType> kokkos_sv_2{kokkos_sv_1};

        CHECK(kokkos_sv_1.getLength() == kokkos_sv_2.getLength());
        CHECK(kokkos_sv_1.getNumQubits() == kokkos_sv_2.getNumQubits());

        std::vector<Kokkos::complex<TestType>> kokkos_sv_1_host(
            kokkos_sv_1.getLength());
        std::vector<Kokkos::complex<TestType>> kokkos_sv_2_host(
            kokkos_sv_2.getLength());
        kokkos_sv_1.DeviceToHost(kokkos_sv_1_host.data(),
                                 kokkos_sv_1.getLength());
        kokkos_sv_2.DeviceToHost(kokkos_sv_2_host.data(),
                                 kokkos_sv_2.getLength());

        for (size_t i = 0; i < kokkos_sv_1_host.size(); i++) {
            CHECK(kokkos_sv_1_host[i] == kokkos_sv_2_host[i]);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyHadamard",
                   "[StateVectorKokkos_Nonparam]", float, double) {

    {
        const std::size_t num_qubits = 3;
        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyHadamard({index}, false);
                Kokkos::complex<TestType> expected(1.0 / std::sqrt(2), 0);
                auto result_subview = Kokkos::subview(kokkos_sv.getData(), 0);
                Kokkos::complex<TestType> result;
                Kokkos::deep_copy(result, result_subview);
                CHECK(expected.real() == Approx(result.real()));
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv(num_qubits);
                kokkos_sv.applyOperation("Hadamard", {index}, false);
                Kokkos::complex<TestType> expected(1.0 / std::sqrt(2), 0);
                auto result_subview = Kokkos::subview(kokkos_sv.getData(), 0);
                Kokkos::complex<TestType> result;
                Kokkos::deep_copy(result, result_subview);
                CHECK(expected.real() == Approx(result.real()));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyPauliX",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        const std::size_t num_qubits = 3;

        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyPauliX({index}, false);

                auto result_subview_0 = Kokkos::subview(kokkos_sv.getData(), 0);
                auto result_subview_1 = Kokkos::subview(
                    kokkos_sv.getData(),
                    0b1 << (kokkos_sv.getNumQubits() - index - 1));
                Kokkos::complex<TestType> result_0, result_1;
                Kokkos::deep_copy(result_0, result_subview_0);
                Kokkos::deep_copy(result_1, result_subview_1);

                CHECK(result_0 == Util::ZERO<Kokkos::complex, TestType>());
                CHECK(result_1 == Util::ONE<Kokkos::complex, TestType>());
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("PauliX", {index}, false);
                auto result_subview_0 = Kokkos::subview(kokkos_sv.getData(), 0);
                auto result_subview_1 = Kokkos::subview(
                    kokkos_sv.getData(),
                    0b1 << (kokkos_sv.getNumQubits() - index - 1));
                Kokkos::complex<TestType> result_0, result_1;
                Kokkos::deep_copy(result_0, result_subview_0);
                Kokkos::deep_copy(result_1, result_subview_1);

                CHECK(result_0 == Util::ZERO<Kokkos::complex, TestType>());
                CHECK(result_1 == Util::ONE<Kokkos::complex, TestType>());
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyPauliY",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});

        const auto p = Util::HALF<Kokkos::complex, TestType>() *
                       Util::INVSQRT2<Kokkos::complex, TestType>() *
                       Util::IMAG<Kokkos::complex, TestType>();
        const auto m = Util::NEGONE<Kokkos::complex, TestType>() * p;

        const std::vector<std::vector<cp_t>> expected_results = {
            {m, m, m, m, p, p, p, p},
            {m, m, p, p, m, m, p, p},
            {m, p, m, p, m, p, m, p}};

        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});

                kokkos_sv.applyPauliY({index}, false);

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getData(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);

                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});

                kokkos_sv.applyOperation("PauliY", {index}, false);
                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getData(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);

                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyPauliZ",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});

        const auto p = Util::HALF<Kokkos::complex, TestType>() *
                       Util::INVSQRT2<Kokkos::complex, TestType>();
        const auto m = Util::NEGONE<Kokkos::complex, TestType>() * p;

        const std::vector<std::vector<cp_t>> expected_results = {
            {p, p, p, p, m, m, m, m},
            {p, p, m, m, p, p, m, m},
            {p, m, p, m, p, m, p, m}};

        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});

                kokkos_sv.applyPauliZ({index}, false);
                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getData(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);

                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});

                kokkos_sv.applyOperation("PauliZ", {index}, false);
                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getData(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);

                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyS", "[StateVectorKokkos_Nonparam]",
                   float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});

        auto r = Util::HALF<Kokkos::complex, TestType>() *
                 Util::INVSQRT2<Kokkos::complex, TestType>();
        auto i = r * Util::IMAG<Kokkos::complex, TestType>();

        const std::vector<std::vector<cp_t>> expected_results = {
            {r, r, r, r, i, i, i, i},
            {r, r, i, i, r, r, i, i},
            {r, i, r, i, r, i, r, i}};

        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});
                kokkos_sv.applyS({index}, false);
                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getData(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);

                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});
                kokkos_sv.applyOperation("S", {index}, false);
                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getData(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);
                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyT", "[StateVectorKokkos_Nonparam]",
                   float, double) {

    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});

        auto r = Util::HALF<Kokkos::complex, TestType>() *
                 Util::INVSQRT2<Kokkos::complex, TestType>();
        auto i = Util::HALF<Kokkos::complex, TestType>() *
                 Util::HALF<Kokkos::complex, TestType>() *
                 (Util::IMAG<Kokkos::complex, TestType>() +
                  Util::ONE<Kokkos::complex, TestType>());

        const std::vector<std::vector<cp_t>> expected_results = {
            {r, r, r, r, i, i, i, i},
            {r, r, i, i, r, r, i, i},
            {r, i, r, i, r, i, r, i}};

        SECTION("Apply directly") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});
                kokkos_sv.applyT({index}, false);
                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getData(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);
                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});
                kokkos_sv.applyOperation("T", {index}, false);

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getData(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);
                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyCNOT",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperation("Hadamard", {0}, false);

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getData());

        auto INVSQRT2 = Util::INVSQRT2<Kokkos::complex, TestType>();

        SECTION("Apply directly") {

            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            Kokkos::deep_copy(kokkos_sv.getData(), ini_sv);

            auto result = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, kokkos_sv.getData());

            for (std::size_t index = 1; index < num_qubits; index++) {
                kokkos_sv.applyCNOT({index - 1, index}, false);
            }

            Kokkos::deep_copy(result, kokkos_sv.getData());
            CHECK(imag(INVSQRT2) == Approx(imag(result[0])));
            CHECK(real(INVSQRT2) == Approx(real(result[0])));
            CHECK(imag(INVSQRT2) == Approx(imag(result[7])));
            CHECK(real(INVSQRT2) == Approx(real(result[7])));
        }
        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            Kokkos::deep_copy(kokkos_sv.getData(), ini_sv);
            auto result = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, kokkos_sv.getData());
            for (std::size_t index = 1; index < num_qubits; index++) {
                kokkos_sv.applyCNOT({index - 1, index}, false);
            }
            Kokkos::deep_copy(result, kokkos_sv.getData());
            CHECK(imag(INVSQRT2) == Approx(imag(result[0])));
            CHECK(real(INVSQRT2) == Approx(real(result[0])));
            CHECK(imag(INVSQRT2) == Approx(imag(result[7])));
            CHECK(real(INVSQRT2) == Approx(real(result[7])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applySWAP",
                   "[StateVectorKokkos_Nonparam]", float, double) {

    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperation({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                 {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getData());

        auto INVSQRT2 = Util::INVSQRT2<Kokkos::complex, TestType>();
        auto ZERO = Util::ZERO<Kokkos::complex, TestType>();

        SECTION("Apply directly") {
            SECTION("Check Initial value") {
                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    if (j == 2) {
                        CHECK(imag(INVSQRT2) == Approx(imag(ini_sv[j])));
                        CHECK(real(INVSQRT2) == Approx(real(ini_sv[j])));
                    } else if (j == 6) {
                        CHECK(imag(INVSQRT2) == Approx(imag(ini_sv[j])));
                        CHECK(real(INVSQRT2) == Approx(real(ini_sv[j])));
                    } else {
                        CHECK(imag(ZERO) == Approx(imag(ini_sv[j])));
                        CHECK(real(ZERO) == Approx(real(ini_sv[j])));
                    }
                }
            }

            SECTION("SWAP0,1 |+10> -> 1+0>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, z, z,
                                                            i, z, i, z};

                StateVectorKokkos<TestType> svdat01{num_qubits};
                StateVectorKokkos<TestType> svdat10{num_qubits};

                Kokkos::deep_copy(svdat01.getData(), ini_sv);
                Kokkos::deep_copy(svdat10.getData(), ini_sv);

                svdat01.applySWAP({0, 1}, false);
                svdat10.applySWAP({1, 0}, false);

                auto sv01 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat01.getData());
                auto sv10 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat10.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {

                    CHECK(imag(expected_results[j]) == Approx(imag(sv01[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv01[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv10[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv10[j])));
                }
            }
            SECTION("SWAP0,2 |+10> -> |01+>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, i,
                                                            z, z, z, z};

                StateVectorKokkos<TestType> svdat02{num_qubits};
                StateVectorKokkos<TestType> svdat20{num_qubits};
                Kokkos::deep_copy(svdat02.getData(), ini_sv);
                Kokkos::deep_copy(svdat20.getData(), ini_sv);

                svdat02.applySWAP({0, 2}, false);
                svdat20.applySWAP({2, 0}, false);

                auto sv02 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat02.getData());
                auto sv20 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat20.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv02[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv02[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv20[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv20[j])));
                }
            }
            SECTION("SWAP1,2 |+10> -> |+01>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, i, z, z,
                                                            z, i, z, z};

                StateVectorKokkos<TestType> svdat12{num_qubits};
                StateVectorKokkos<TestType> svdat21{num_qubits};
                Kokkos::deep_copy(svdat12.getData(), ini_sv);
                Kokkos::deep_copy(svdat21.getData(), ini_sv);

                svdat12.applySWAP({1, 2}, false);
                svdat21.applySWAP({2, 1}, false);

                auto sv12 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat12.getData());
                auto sv21 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat21.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv12[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv12[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv21[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv21[j])));
                }
            }
        }
        SECTION("Apply using dispatcher") {
            SECTION("SWAP0,1 |+10> -> 1+0>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, z, z,
                                                            i, z, i, z};

                StateVectorKokkos<TestType> svdat01{num_qubits};
                StateVectorKokkos<TestType> svdat10{num_qubits};
                Kokkos::deep_copy(svdat01.getData(), ini_sv);
                Kokkos::deep_copy(svdat10.getData(), ini_sv);

                svdat01.applyOperation("SWAP", {0, 1}, false);
                svdat10.applyOperation("SWAP", {1, 0}, false);

                auto sv01 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat01.getData());
                auto sv10 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat10.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv01[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv01[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv10[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv10[j])));
                }
            }
            SECTION("SWAP0,2 |+10> -> |01+>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, i,
                                                            z, z, z, z};

                StateVectorKokkos<TestType> svdat02{num_qubits};
                StateVectorKokkos<TestType> svdat20{num_qubits};
                Kokkos::deep_copy(svdat02.getData(), ini_sv);
                Kokkos::deep_copy(svdat20.getData(), ini_sv);

                svdat02.applyOperation("SWAP", {0, 2}, false);
                svdat20.applyOperation("SWAP", {2, 0}, false);

                auto sv02 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat02.getData());
                auto sv20 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat20.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv02[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv02[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv20[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv20[j])));
                }
            }
            SECTION("SWAP1,2 |+10> -> |+01>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, i, z, z,
                                                            z, i, z, z};

                StateVectorKokkos<TestType> svdat12{num_qubits};
                StateVectorKokkos<TestType> svdat21{num_qubits};
                Kokkos::deep_copy(svdat12.getData(), ini_sv);
                Kokkos::deep_copy(svdat21.getData(), ini_sv);

                svdat12.applyOperation("SWAP", {1, 2}, false);
                svdat21.applyOperation("SWAP", {2, 1}, false);

                auto sv12 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat12.getData());
                auto sv21 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat21.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv12[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv12[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv21[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv21[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyCZ", "[StateVectorKokkos_Nonparam]",
                   float, double) {

    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperation({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                 {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getData());

        auto INVSQRT2 = Util::INVSQRT2<Kokkos::complex, TestType>();
        auto ZERO = Util::ZERO<Kokkos::complex, TestType>();

        SECTION("Apply directly") {
            SECTION("Check Initial value") {
                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    if (j == 2) {
                        CHECK(imag(INVSQRT2) == Approx(imag(ini_sv[j])));
                        CHECK(real(INVSQRT2) == Approx(real(ini_sv[j])));
                    } else if (j == 6) {
                        CHECK(imag(INVSQRT2) == Approx(imag(ini_sv[j])));
                        CHECK(real(INVSQRT2) == Approx(real(ini_sv[j])));
                    } else {
                        CHECK(imag(ZERO) == Approx(imag(ini_sv[j])));
                        CHECK(real(ZERO) == Approx(real(ini_sv[j])));
                    }
                }
            }

            SECTION("CZ0,1 |+10> -> 1+0>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i,  z,
                                                            z, z, -i, z};

                StateVectorKokkos<TestType> svdat01{num_qubits};
                StateVectorKokkos<TestType> svdat10{num_qubits};

                Kokkos::deep_copy(svdat01.getData(), ini_sv);
                Kokkos::deep_copy(svdat10.getData(), ini_sv);

                svdat01.applyCZ({0, 1}, false);
                svdat10.applyCZ({1, 0}, false);

                auto sv01 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat01.getData());
                auto sv10 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat10.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv01[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv01[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv10[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv10[j])));
                }
            }
            SECTION("CZ0,2 |+10> -> |01+>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, z,
                                                            z, z, i, z};

                StateVectorKokkos<TestType> svdat02{num_qubits};
                StateVectorKokkos<TestType> svdat20{num_qubits};
                Kokkos::deep_copy(svdat02.getData(), ini_sv);
                Kokkos::deep_copy(svdat20.getData(), ini_sv);

                svdat02.applyCZ({0, 2}, false);
                svdat20.applyCZ({2, 0}, false);

                auto sv02 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat02.getData());
                auto sv20 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat20.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv02[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv02[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv20[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv20[j])));
                }
            }
            SECTION("CZ1,2 |+10> -> |+01>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, z,
                                                            z, z, i, z};

                StateVectorKokkos<TestType> svdat12{num_qubits};
                StateVectorKokkos<TestType> svdat21{num_qubits};
                Kokkos::deep_copy(svdat12.getData(), ini_sv);
                Kokkos::deep_copy(svdat21.getData(), ini_sv);

                svdat12.applyCZ({1, 2}, false);
                svdat21.applyCZ({2, 1}, false);

                auto sv12 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat12.getData());
                auto sv21 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat21.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv12[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv12[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv21[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv21[j])));
                }
            }
        }
        SECTION("Apply using dispatcher") {
            SECTION("CZ0,1 |+10> -> 1+0>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i,  z,
                                                            z, z, -i, z};

                StateVectorKokkos<TestType> svdat01{num_qubits};
                StateVectorKokkos<TestType> svdat10{num_qubits};
                Kokkos::deep_copy(svdat01.getData(), ini_sv);
                Kokkos::deep_copy(svdat10.getData(), ini_sv);

                svdat01.applyOperation("CZ", {0, 1}, false);
                svdat10.applyOperation("CZ", {1, 0}, false);

                auto sv01 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat01.getData());
                auto sv10 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat10.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv01[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv01[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv10[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv10[j])));
                }
            }
            SECTION("CZ0,2 |+10> -> |01+>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, z,
                                                            z, z, i, z};

                StateVectorKokkos<TestType> svdat02{num_qubits};
                StateVectorKokkos<TestType> svdat20{num_qubits};
                Kokkos::deep_copy(svdat02.getData(), ini_sv);
                Kokkos::deep_copy(svdat20.getData(), ini_sv);

                svdat02.applyOperation("CZ", {0, 2}, false);
                svdat20.applyOperation("CZ", {2, 0}, false);

                auto sv02 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat02.getData());
                auto sv20 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat20.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv02[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv02[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv20[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv20[j])));
                }
            }
            SECTION("CZ1,2 |+10> -> |+01>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, z,
                                                            z, z, i, z};

                StateVectorKokkos<TestType> svdat12{num_qubits};
                StateVectorKokkos<TestType> svdat21{num_qubits};
                Kokkos::deep_copy(svdat12.getData(), ini_sv);
                Kokkos::deep_copy(svdat21.getData(), ini_sv);

                svdat12.applyOperation("CZ", {1, 2}, false);
                svdat21.applyOperation("CZ", {2, 1}, false);

                auto sv12 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat12.getData());
                auto sv21 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat21.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv12[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv12[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv21[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv21[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyToffoli",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperation({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                 {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getData());

        auto ZERO = Util::ZERO<Kokkos::complex, TestType>();
        auto INVSQRT2 = Util::INVSQRT2<Kokkos::complex, TestType>();

        SECTION("Apply directly") {

            SECTION("Toffoli 0,1,2 |+10> -> 010> + 111>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, z,
                                                            z, z, z, i};

                StateVectorKokkos<TestType> svdat012{num_qubits};

                Kokkos::deep_copy(svdat012.getData(), ini_sv);

                svdat012.applyToffoli({0, 1, 2}, false);

                auto sv012 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat012.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv012[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv012[j])));
                }
            }
            SECTION("Toffoli 1,0,2 |+10> -> |010> + |111>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, z,
                                                            z, z, z, i};

                StateVectorKokkos<TestType> svdat102{num_qubits};
                Kokkos::deep_copy(svdat102.getData(), ini_sv);

                svdat102.applyToffoli({1, 0, 2}, false);

                auto sv102 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat102.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv102[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv102[j])));
                }
            }
            SECTION("Toffoli 1,2,0 |+10> -> |+10>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, z,
                                                            z, z, i, z};

                StateVectorKokkos<TestType> svdat120{num_qubits};
                Kokkos::deep_copy(svdat120.getData(), ini_sv);

                svdat120.applyToffoli({1, 2, 0}, false);

                auto sv120 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat120.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv120[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv120[j])));
                }
            }
        }
        SECTION("Apply using dispatcher") {
            SECTION("Toffoli [0,1,2],[1,0,2] |+10> -> +1+>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, z,
                                                            z, z, z, i};

                StateVectorKokkos<TestType> svdat012{num_qubits};
                StateVectorKokkos<TestType> svdat102{num_qubits};
                Kokkos::deep_copy(svdat012.getData(), ini_sv);
                Kokkos::deep_copy(svdat102.getData(), ini_sv);

                svdat012.applyOperation("Toffoli", {0, 1, 2}, false);
                svdat102.applyOperation("Toffoli", {1, 0, 2}, false);

                auto sv012 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat012.getData());
                auto sv102 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat102.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv012[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv012[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv102[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv102[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyMultiQubitOp",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    size_t num_qubits = 3;
    StateVectorKokkos<TestType> sv_normal{num_qubits};
    StateVectorKokkos<TestType> sv_mq{num_qubits};
    using UnmanagedComplexHostView =
        Kokkos::View<Kokkos::complex<TestType> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    SECTION("Single Qubit") {
        auto matrix = getHadamard<Kokkos::complex, TestType>();
        std::vector<size_t> wires = {0};
        sv_normal.applyOperation("Hadamard", wires, false);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getData());

        Kokkos::View<Kokkos::complex<TestType> *> device_matrix("device_matrix",
                                                                matrix.size());
        Kokkos::deep_copy(device_matrix, UnmanagedComplexHostView(
                                             matrix.data(), matrix.size()));
        sv_mq.applyMultiQubitOp(device_matrix, wires, false);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getData());

        for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }

    SECTION("Two Qubit") {
        auto matrix = getCNOT<Kokkos::complex, TestType>();
        std::vector<size_t> wires = {0, 1};
        sv_normal.applyOperation("CNOT", wires, false);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getData());

        Kokkos::View<Kokkos::complex<TestType> *> device_matrix("device_matrix",
                                                                matrix.size());
        Kokkos::deep_copy(device_matrix, UnmanagedComplexHostView(
                                             matrix.data(), matrix.size()));
        sv_mq.applyMultiQubitOp(device_matrix, wires, false);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getData());

        for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }

    SECTION("Three Qubit") {
        auto matrix = getToffoli<Kokkos::complex, TestType>();
        std::vector<size_t> wires = {0, 1, 2};
        sv_normal.applyOperation("Toffoli", wires, false);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getData());

        Kokkos::View<Kokkos::complex<TestType> *> device_matrix("device_matrix",
                                                                matrix.size());
        Kokkos::deep_copy(device_matrix, UnmanagedComplexHostView(
                                             matrix.data(), matrix.size()));
        sv_mq.applyMultiQubitOp(device_matrix, wires, false);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getData());

        for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyCSWAP",
                   "[StateVectorKokkos_Nonparam]", float, double) {

    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperation({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                 {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getData());

        auto ZERO = Util::ZERO<Kokkos::complex, TestType>();
        auto INVSQRT2 = Util::INVSQRT2<Kokkos::complex, TestType>();

        SECTION("Apply directly") {

            SECTION("CSWAP 0,1,2 |+10> -> 010> + 111>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, z,
                                                            z, i, z, z};

                StateVectorKokkos<TestType> svdat012{num_qubits};

                Kokkos::deep_copy(svdat012.getData(), ini_sv);

                svdat012.applyCSWAP({0, 1, 2}, false);

                auto sv012 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat012.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv012[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv012[j])));
                }
            }
            SECTION("CSWAP 1,0,2 |+10> -> |010> + |111>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, i,
                                                            z, z, z, z};

                StateVectorKokkos<TestType> svdat102{num_qubits};
                Kokkos::deep_copy(svdat102.getData(), ini_sv);

                svdat102.applyCSWAP({1, 0, 2}, false);

                auto sv102 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat102.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv102[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv102[j])));
                }
            }
            SECTION("CSWAP 2,1,0 |+10> -> |+10>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, z,
                                                            z, z, i, z};

                StateVectorKokkos<TestType> svdat210{num_qubits};
                Kokkos::deep_copy(svdat210.getData(), ini_sv);

                svdat210.applyCSWAP({2, 1, 0}, false);

                auto sv210 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat210.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv210[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv210[j])));
                }
            }
        }
        SECTION("Apply using dispatcher") {
            SECTION("CSWAP [0,1,2]|+10> -> |010> + |101>") {
                auto z = ZERO;
                auto i = INVSQRT2;
                const std::vector<cp_t> expected_results = {z, z, i, z,
                                                            z, i, z, z};

                StateVectorKokkos<TestType> svdat012{num_qubits};
                Kokkos::deep_copy(svdat012.getData(), ini_sv);

                svdat012.applyOperation("CSWAP", {0, 1, 2}, false);

                auto sv012 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat012.getData());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv012[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv012[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::SetStateVector",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using cp_t = Kokkos::complex<TestType>;
    const std::size_t num_qubits = 3;

    //`values[i]` on the host will be copy the `indices[i]`th element of the
    // state vector on the device.
    SECTION("Set state vector with values and their corresponding indices on "
            "the host") {
        std::vector<cp_t> init_state{
            cp_t{0.267462849617, 0.010768564418},
            cp_t{0.228575125337, 0.010564590804},
            cp_t{0.099492751062, 0.260849833488},
            cp_t{0.093690201640, 0.189847111702},
            cp_t{0.015641822883, 0.225092900621},
            cp_t{0.205574608177, 0.082808663337},
            cp_t{0.006827173322, 0.211631480575},
            cp_t{0.255280800811, 0.161572331669},
        };
        auto expected_state = init_state;

        for (size_t i = 0;
             i < Pennylane::Lightning_Kokkos::Util::exp2(num_qubits - 1); i++) {
            std::swap(expected_state[i * 2], expected_state[i * 2 + 1]);
        }

        StateVectorKokkos<PrecisionT> kokkos_sv{num_qubits};
        std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
        kokkos_sv.HostToDevice(init_state.data(), init_state.size());

        // The setStates will shuffle the state vector values on the device with
        // the following indices and values setting on host. For example, the
        // values[i] is used to set the indices[i] th element of state vector on
        // the device. For example, values[2] (init_state[5]) will be copied to
        // indices[2]th or (4th) element of the state vector.
        std::vector<std::size_t> indices = {0, 2, 4, 6, 1, 3, 5, 7};

        std::vector<Kokkos::complex<PrecisionT>> values = {
            init_state[1], init_state[3], init_state[5], init_state[7],
            init_state[0], init_state[2], init_state[4], init_state[6]};

        kokkos_sv.setStateVector(indices, values);

        kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

        for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
            CHECK(imag(expected_state[j]) == Approx(imag(result_sv[j])));
            CHECK(real(expected_state[j]) == Approx(real(result_sv[j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::SetIthStates",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using cp_t = Kokkos::complex<TestType>;
    const std::size_t num_qubits = 3;

    SECTION(
        "Set Ith element of the state state on device with data on the host") {
        std::vector<cp_t> init_state{
            cp_t{0.267462849617, 0.010768564418},
            cp_t{0.228575125337, 0.010564590804},
            cp_t{0.099492751062, 0.260849833488},
            cp_t{0.093690201640, 0.189847111702},
            cp_t{0.015641822883, 0.225092900621},
            cp_t{0.205574608177, 0.082808663337},
            cp_t{0.006827173322, 0.211631480575},
            cp_t{0.255280800811, 0.161572331669},
        };

        std::vector<cp_t> expected_state{
            cp_t{0.0, 0.0}, cp_t{0.0, 0.0}, cp_t{0.0, 0.0}, cp_t{1.0, 0.0},
            cp_t{0.0, 0.0}, cp_t{0.0, 0.0}, cp_t{0.0, 0.0}, cp_t{0.0, 0.0},
        };

        StateVectorKokkos<PrecisionT> kokkos_sv{num_qubits};
        std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
        kokkos_sv.HostToDevice(init_state.data(), init_state.size());

        size_t index = 3;

        kokkos_sv.setBasisState(index);

        kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

        for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
            CHECK(imag(expected_state[j]) == Approx(imag(result_sv[j])));
            CHECK(real(expected_state[j]) == Approx(real(result_sv[j])));
        }
    }
}
