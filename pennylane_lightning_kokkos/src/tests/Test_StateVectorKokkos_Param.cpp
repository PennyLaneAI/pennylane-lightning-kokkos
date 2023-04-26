#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "GatesHost.hpp"
#include "MeasuresKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"

using namespace Pennylane;

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyIsingXY",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;

        std::vector<cp_t> ini_st{
            cp_t{0.267462841882, 0.010768564798},
            cp_t{0.228575129706, 0.010564590956},
            cp_t{0.099492749900, 0.260849823392},
            cp_t{0.093690204310, 0.189847108173},
            cp_t{0.033390732374, 0.203836830144},
            cp_t{0.226979395737, 0.081852150975},
            cp_t{0.031235505729, 0.176933497281},
            cp_t{0.294287602843, 0.145156781198},
            cp_t{0.152742706049, 0.111628061129},
            cp_t{0.012553863703, 0.120027860480},
            cp_t{0.237156555364, 0.154658769755},
            cp_t{0.117001120872, 0.228059505033},
            cp_t{0.041495873225, 0.065934827444},
            cp_t{0.089653239407, 0.221581340372},
            cp_t{0.217892322429, 0.291261296999},
            cp_t{0.292993251871, 0.186570798697},
        };

        std::vector<cp_t> expected{
            cp_t{0.267462849617, 0.010768564418},
            cp_t{0.228575125337, 0.010564590804},
            cp_t{0.099492751062, 0.260849833488},
            cp_t{0.093690201640, 0.189847111702},
            cp_t{0.015641822883, 0.225092900621},
            cp_t{0.205574608177, 0.082808663337},
            cp_t{0.006827173322, 0.211631480575},
            cp_t{0.255280800811, 0.161572331669},
            cp_t{0.119218164572, 0.115460377284},
            cp_t{-0.000315789761, 0.153835664378},
            cp_t{0.206786872079, 0.157633689097},
            cp_t{0.093027614553, 0.271012980118},
            cp_t{0.041495874524, 0.065934829414},
            cp_t{0.089653238654, 0.221581339836},
            cp_t{0.217892318964, 0.291261285543},
            cp_t{0.292993247509, 0.186570793390},
        };

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyIsingXY({0, 1}, false, {0.312});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("IsingXY", {0, 1}, false, {0.312});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyRX",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        size_t num_qubits = 1;
        std::vector<TestType> angles = {0.1, 0.6};
        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>{cp_t{0.9987502603949663, 0.0},
                              cp_t{0.0, -0.04997916927067834}},
            std::vector<cp_t>{cp_t{0.9553364891256061, 0.0},
                              cp_t{0, -0.2955202066613395}},
            std::vector<cp_t>{cp_t{0.49757104789172696, 0.0},
                              cp_t{0, -0.867423225594017}}};

        std::vector<std::vector<cp_t>> expected_results_adj{
            std::vector<cp_t>{cp_t{0.9987502603949663, 0.0},
                              cp_t{0.0, 0.04997916927067834}},
            std::vector<cp_t>{cp_t{0.9553364891256061, 0.0},
                              cp_t{0, 0.2955202066613395}},
            std::vector<cp_t>{cp_t{0.49757104789172696, 0.0},
                              cp_t{0, 0.867423225594017}}};

        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

                kokkos_sv.applyRX({0}, false, {angles[index]});
                kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result_sv[j])));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result_sv[j])));
                }
            }
        }

        SECTION("Apply adj directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

                kokkos_sv.applyRX({0}, true, {angles[index]});
                kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results_adj[index][j]) ==
                          Approx(imag(result_sv[j])));
                    CHECK(real(expected_results_adj[index][j]) ==
                          Approx(real(result_sv[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyRY",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        size_t num_qubits = 1;
        const std::vector<TestType> angles{0.2, 0.7, 2.9};
        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>{{0.8731983044562817, 0.04786268954660339},
                              {0.0876120655431924, -0.47703040785184303}},
            std::vector<cp_t>{{0.8243771119105122, 0.16439396602553008},
                              {0.3009211363333468, -0.45035926880694604}},
            std::vector<cp_t>{{0.10575112905629831, 0.47593196040758534},
                              {0.8711876098966215, -0.0577721051072477}}};
        std::vector<std::vector<cp_t>> expected_results_adj{
            std::vector<cp_t>{{0.8731983044562817, -0.04786268954660339},
                              {-0.0876120655431924, -0.47703040785184303}},
            std::vector<cp_t>{{0.8243771119105122, -0.16439396602553008},
                              {-0.3009211363333468, -0.45035926880694604}},
            std::vector<cp_t>{{0.10575112905629831, -0.47593196040758534},
                              {-0.8711876098966215, -0.0577721051072477}}};

        std::vector<cp_t> ini_st{{0.8775825618903728, 0.0},
                                 {0.0, -0.47942553860420306}};

        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

                kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
                kokkos_sv.applyRY({0}, false, {angles[index]});
                kokkos_sv.DeviceToHost(result_sv.data(), ini_st.size());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result_sv[j])));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result_sv[j])));
                }
            }
        }

        SECTION("Apply adj directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

                kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
                kokkos_sv.applyRY({0}, true, {angles[index]});
                kokkos_sv.DeviceToHost(result_sv.data(), ini_st.size());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK(imag(expected_results_adj[index][j]) ==
                          Approx(imag(result_sv[j])));
                    CHECK(real(expected_results_adj[index][j]) ==
                          Approx(real(result_sv[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyRZ",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        size_t num_qubits = 3;
        const std::vector<TestType> angles{0.2, 0.7, 2.9};
        const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

        std::vector<std::vector<cp_t>> rz_data;

        rz_data.reserve(angles.size());
        for (auto &a : angles) {
            rz_data.push_back(Gates::getRZ<Kokkos::complex, TestType>(a));
        }

        std::vector<std::vector<cp_t>> expected_results = {
            {rz_data[0][0], rz_data[0][0], rz_data[0][0], rz_data[0][0],
             rz_data[0][3], rz_data[0][3], rz_data[0][3], rz_data[0][3]},
            {
                rz_data[1][0],
                rz_data[1][0],
                rz_data[1][3],
                rz_data[1][3],
                rz_data[1][0],
                rz_data[1][0],
                rz_data[1][3],
                rz_data[1][3],
            },
            {rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3],
             rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3]}};

        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});
                kokkos_sv.applyRZ({index}, false, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j]) * 2 * std::sqrt(2)) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j]) * 2 * std::sqrt(2)) ==
                          Approx(imag(expected_results[index][j])));
                }
            }
        }

        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});
                kokkos_sv.applyOperation("RZ", {index}, false, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j]) * 2 * std::sqrt(2)) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j]) * 2 * std::sqrt(2)) ==
                          Approx(imag(expected_results[index][j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyPhaseShift",
                   "[StateVectorKokkosManaged_Param]", double) {
    using cp_t = Kokkos::complex<TestType>;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8, 2.4};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<cp_t>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(Gates::getPhaseShift<Kokkos::complex, TestType>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][3], ps_data[0][3], ps_data[0][3], ps_data[0][3]},
        {
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
        },
        {ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3],
         ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3]}};

    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {

            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                     {{0}, {1}, {2}},
                                     {{false}, {false}, {false}});
            kokkos_sv.applyPhaseShift({index}, false, {angles[index]});
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK((real(result_sv[j]) * 2 * std::sqrt(2)) ==
                      Approx(real(expected_results[index][j])));
                CHECK((imag(result_sv[j]) * 2 * std::sqrt(2)) ==
                      Approx(imag(expected_results[index][j])));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                     {{0}, {1}, {2}},
                                     {{false}, {false}, {false}});
            kokkos_sv.applyOperation("PhaseShift", {index}, false,
                                     {angles[index]});
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK((real(result_sv[j]) * 2 * std::sqrt(2)) ==
                      Approx(real(expected_results[index][j])));
                CHECK((imag(result_sv[j]) * 2 * std::sqrt(2)) ==
                      Approx(imag(expected_results[index][j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyControlledPhaseShift",
                   "[StateVectorKokkosManaged_Param]", double) {
    using cp_t = Kokkos::complex<TestType>;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 2.4};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<cp_t>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(Gates::getPhaseShift<Kokkos::complex, TestType>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][0], ps_data[0][0], ps_data[0][3], ps_data[0][3]},
        {ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3],
         ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3]}};

    SECTION("Apply directly") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        kokkos_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});
        kokkos_sv.applyControlledPhaseShift({0, 1}, false, {angles[0]});
        std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
        kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

        for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
            CHECK((real(result_sv[j]) * 2 * std::sqrt(2)) ==
                  Approx(real(expected_results[0][j])));
            CHECK((imag(result_sv[j]) * 2 * std::sqrt(2)) ==
                  Approx(imag(expected_results[0][j])));
        }
    }
    SECTION("Apply using dispatcher") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        kokkos_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});
        kokkos_sv.applyOperation("ControlledPhaseShift", {1, 2}, false,
                                 {angles[1]});
        std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
        kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

        for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
            CHECK((real(result_sv[j]) * 2 * std::sqrt(2)) ==
                  Approx(real(expected_results[1][j])));
            CHECK((imag(result_sv[j]) * 2 * std::sqrt(2)) ==
                  Approx(imag(expected_results[1][j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyRot",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using cp_t = Kokkos::complex<TestType>;
    const size_t num_qubits = 3;
    const std::vector<std::vector<TestType>> angles{
        std::vector<TestType>{0.3, 0.8, 2.4},
        std::vector<TestType>{0.5, 1.1, 3.0},
        std::vector<TestType>{2.3, 0.1, 0.4}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto rot_mat = Gates::getRot<Kokkos::complex, TestType>(
            angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = rot_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rot_mat[2];
    }

    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};

            kokkos_sv.applyRot({index}, false, angles[index]);
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK((real(result_sv[j])) ==
                      Approx(real(expected_results[index][j])));
                CHECK((imag(result_sv[j])) ==
                      Approx(imag(expected_results[index][j])));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};

            kokkos_sv.applyOperation("Rot", {index}, false, angles[index]);
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK((real(result_sv[j])) ==
                      Approx(real(expected_results[index][j])));
                CHECK((imag(result_sv[j])) ==
                      Approx(imag(expected_results[index][j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyCRot",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using cp_t = Kokkos::complex<TestType>;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8, 2.4};
    std::vector<cp_t> expected_results(8);
    const auto rot_mat = Gates::getRot<Kokkos::complex, TestType>(
        angles[0], angles[1], angles[2]);
    expected_results[0b1 << (num_qubits - 1)] = rot_mat[0];
    expected_results[(0b1 << num_qubits) - 2] = rot_mat[2];

    SECTION("Apply directly") {

        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation({"PauliX"}, {0}, false);
            kokkos_sv.applyCRot({0, 1}, false, angles);
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK((real(result_sv[j])) ==
                      Approx(real(expected_results[j])));
                CHECK((imag(result_sv[j])) ==
                      Approx(imag(expected_results[j])));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation("PauliX", {0}, false);
            kokkos_sv.applyOperation("CRot", {0, 1}, false, angles);
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK((real(result_sv[j])) ==
                      Approx(real(expected_results[j])));
                CHECK((imag(result_sv[j])) ==
                      Approx(imag(expected_results[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyIsingXX",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using cp_t = Kokkos::complex<TestType>;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = cp_t{0.9887710779360422, 0.0};
    expected_results[0][6] = cp_t{0.0, -0.14943813247359922};

    expected_results[1][0] = cp_t{0.9210609940028851, 0.0};
    expected_results[1][6] = cp_t{0.0, -0.3894183423086505};

    expected_results[2][0] = cp_t{0.9887710779360422, 0.0};
    expected_results[2][5] = cp_t{0.0, -0.14943813247359922};

    expected_results[3][0] = cp_t{0.9210609940028851, 0.0};
    expected_results[3][5] = cp_t{0.0, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};

    expected_results_adj[0][0] = cp_t{0.9887710779360422, 0.0};
    expected_results_adj[0][6] = cp_t{0.0, 0.14943813247359922};

    expected_results_adj[1][0] = cp_t{0.9210609940028851, 0.0};
    expected_results_adj[1][6] = cp_t{0.0, 0.3894183423086505};

    expected_results_adj[2][0] = cp_t{0.9887710779360422, 0.0};
    expected_results_adj[2][5] = cp_t{0.0, 0.14943813247359922};

    expected_results_adj[3][0] = cp_t{0.9210609940028851, 0.0};
    expected_results_adj[3][5] = cp_t{0.0, 0.3894183423086505};

    SECTION("Apply directly adjoint=false") {
        SECTION("IsingXX 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingXX({0, 1}, false, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results[index][j])));
                }
            }
        }
        SECTION("IsingXX 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {

                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingXX({0, 2}, false, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(
                              expected_results[index + angles.size()][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(
                              expected_results[index + angles.size()][j])));
                }
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("IsingXX 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingXX({0, 1}, true, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results_adj[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results_adj[index][j])));
                }
            }
        }
        SECTION("IsingXX 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingXX({0, 2}, true, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(
                              expected_results_adj[index + angles.size()][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(
                              expected_results_adj[index + angles.size()][j])));
                }
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation("IsingXX", {0, 1}, true, {angles[index]});
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK((real(result_sv[j])) ==
                      Approx(real(expected_results_adj[index][j])));
                CHECK((imag(result_sv[j])) ==
                      Approx(imag(expected_results_adj[index][j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyIsingYY",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using cp_t = Kokkos::complex<TestType>;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = cp_t{0.9887710779360422, 0.0};
    expected_results[0][6] = cp_t{0.0, 0.14943813247359922};

    expected_results[1][0] = cp_t{0.9210609940028851, 0.0};
    expected_results[1][6] = cp_t{0.0, 0.3894183423086505};

    expected_results[2][0] = cp_t{0.9887710779360422, 0.0};
    expected_results[2][5] = cp_t{0.0, 0.14943813247359922};

    expected_results[3][0] = cp_t{0.9210609940028851, 0.0};
    expected_results[3][5] = cp_t{0.0, 0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};

    expected_results_adj[0][0] = cp_t{0.9887710779360422, 0.0};
    expected_results_adj[0][6] = cp_t{0.0, -0.14943813247359922};

    expected_results_adj[1][0] = cp_t{0.9210609940028851, 0.0};
    expected_results_adj[1][6] = cp_t{0.0, -0.3894183423086505};

    expected_results_adj[2][0] = cp_t{0.9887710779360422, 0.0};
    expected_results_adj[2][5] = cp_t{0.0, -0.14943813247359922};

    expected_results_adj[3][0] = cp_t{0.9210609940028851, 0.0};
    expected_results_adj[3][5] = cp_t{0.0, -0.3894183423086505};

    SECTION("Apply directly adjoint=false") {
        SECTION("IsingYY 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingYY({0, 1}, false, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results[index][j])));
                }
            }
        }
        SECTION("IsingYY 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {

                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingYY({0, 2}, false, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(
                              expected_results[index + angles.size()][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(
                              expected_results[index + angles.size()][j])));
                }
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("IsingYY 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingYY({0, 1}, true, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results_adj[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results_adj[index][j])));
                }
            }
        }
        SECTION("IsingYY 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingYY({0, 2}, true, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(
                              expected_results_adj[index + angles.size()][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(
                              expected_results_adj[index + angles.size()][j])));
                }
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation("IsingYY", {0, 1}, true, {angles[index]});
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK((real(result_sv[j])) ==
                      Approx(real(expected_results_adj[index][j])));
                CHECK((imag(result_sv[j])) ==
                      Approx(imag(expected_results_adj[index][j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyIsingZZ",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using cp_t = Kokkos::complex<TestType>;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0, 0}),
        std::vector<cp_t>(1 << num_qubits, {0, 0})};
    expected_results[0][0] = cp_t{0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = cp_t{0.9210609940028851, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = cp_t{0.9887710779360422, 0.14943813247359922};
    expected_results_adj[1][0] = cp_t{0.9210609940028851, 0.3894183423086505};

    SECTION("Apply directly adjoint=false") {
        SECTION("IsingZZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingZZ({0, 1}, false, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results[index][j])));
                }
            }
        }
        SECTION("IsingZZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {

                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingZZ({0, 2}, false, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results[index][j])));
                }
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("IsingZZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingZZ({0, 1}, true, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results_adj[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results_adj[index][j])));
                }
            }
        }
        SECTION("IsingZZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyIsingZZ({0, 2}, true, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results_adj[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results_adj[index][j])));
                }
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation("IsingZZ", {0, 1}, true, {angles[index]});
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK((real(result_sv[j])) ==
                      Approx(real(expected_results_adj[index][j])));
                CHECK((imag(result_sv[j])) ==
                      Approx(imag(expected_results_adj[index][j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyMultiRZ",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using cp_t = Kokkos::complex<TestType>;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = cp_t{0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = cp_t{0.9210609940028851, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = cp_t{0.9887710779360422, 0.14943813247359922};
    expected_results_adj[1][0] = cp_t{0.9210609940028851, 0.3894183423086505};

    SECTION("Apply directly adjoint=false") {
        SECTION("MultiRZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyMultiRZ({0, 1}, false, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results[index][j])));
                }
            }
        }
        SECTION("MultiRZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyMultiRZ({0, 2}, false, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results[index][j])));
                }
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("MultiRZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyMultiRZ({0, 1}, true, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results_adj[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results_adj[index][j])));
                }
            }
        }
        SECTION("MultiRZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyMultiRZ({0, 2}, true, {angles[index]});
                std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results_adj[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(imag(expected_results_adj[index][j])));
                }
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation("MultiRZ", {0, 1}, true, {angles[index]});
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK((real(result_sv[j])) ==
                      Approx(real(expected_results_adj[index][j])));
                CHECK((imag(result_sv[j])) ==
                      Approx(imag(expected_results_adj[index][j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applySingleExcitation",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 3;

        std::vector<cp_t> ini_st{
            cp_t{0.125681356503, 0.252712197380},
            cp_t{0.262591068130, 0.370189000494},
            cp_t{0.129300299863, 0.371057794075},
            cp_t{0.392248682814, 0.195795523118},
            cp_t{0.303908059240, 0.082981563244},
            cp_t{0.189140284321, 0.179512645957},
            cp_t{0.173146612336, 0.092249594834},
            cp_t{0.298857179897, 0.269627836165},
        };

        std::vector<cp_t> expected{
            cp_t{0.125681, 0.252712}, cp_t{0.219798, 0.355848},
            cp_t{0.1293, 0.371058},   cp_t{0.365709, 0.181773},
            cp_t{0.336159, 0.131522}, cp_t{0.18914, 0.179513},
            cp_t{0.223821, 0.117493}, cp_t{0.298857, 0.269628},
        };

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applySingleExcitation({0, 2}, false, {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("SingleExcitation", {0, 2}, false,
                                     {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applySingleExcitationMinus",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 3;

        std::vector<cp_t> ini_st{
            cp_t{0.125681356503, 0.252712197380},
            cp_t{0.262591068130, 0.370189000494},
            cp_t{0.129300299863, 0.371057794075},
            cp_t{0.392248682814, 0.195795523118},
            cp_t{0.303908059240, 0.082981563244},
            cp_t{0.189140284321, 0.179512645957},
            cp_t{0.173146612336, 0.092249594834},
            cp_t{0.298857179897, 0.269627836165},
        };

        std::vector<cp_t> expected{
            cp_t{0.158204, 0.233733}, cp_t{0.219798, 0.355848},
            cp_t{0.177544, 0.350543}, cp_t{0.365709, 0.181773},
            cp_t{0.336159, 0.131522}, cp_t{0.211353, 0.152737},
            cp_t{0.223821, 0.117493}, cp_t{0.33209, 0.227445},
        };

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applySingleExcitationMinus({0, 2}, false,
                                                 {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("SingleExcitationMinus", {0, 2}, false,
                                     {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applySingleExcitationPlus",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 3;

        std::vector<cp_t> ini_st{
            cp_t{0.125681356503, 0.252712197380},
            cp_t{0.262591068130, 0.370189000494},
            cp_t{0.129300299863, 0.371057794075},
            cp_t{0.392248682814, 0.195795523118},
            cp_t{0.303908059240, 0.082981563244},
            cp_t{0.189140284321, 0.179512645957},
            cp_t{0.173146612336, 0.092249594834},
            cp_t{0.298857179897, 0.269627836165},
        };

        std::vector<cp_t> expected{
            cp_t{0.090922, 0.267194},  cp_t{0.219798, 0.355848},
            cp_t{0.0787548, 0.384968}, cp_t{0.365709, 0.181773},
            cp_t{0.336159, 0.131522},  cp_t{0.16356, 0.203093},
            cp_t{0.223821, 0.117493},  cp_t{0.260305, 0.307012},
        };

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applySingleExcitationPlus({0, 2}, false,
                                                {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("SingleExcitationPlus", {0, 2}, false,
                                     {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyDoubleExcitation",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;

        std::vector<cp_t> ini_st{
            cp_t{0.125681356503, 0.252712197380},
            cp_t{0.262591068130, 0.370189000494},
            cp_t{0.129300299863, 0.371057794075},
            cp_t{0.392248682814, 0.195795523118},
            cp_t{0.303908059240, 0.082981563244},
            cp_t{0.189140284321, 0.179512645957},
            cp_t{0.173146612336, 0.092249594834},
            cp_t{0.298857179897, 0.269627836165},
            cp_t{0.125681356503, 0.252712197380},
            cp_t{0.262591068130, 0.370189000494},
            cp_t{0.129300299863, 0.371057794075},
            cp_t{0.392248682814, 0.195795523118},
            cp_t{0.303908059240, 0.082981563244},
            cp_t{0.189140284321, 0.179512645957},
            cp_t{0.173146612336, 0.092249594834},
            cp_t{0.298857179897, 0.269627836165},
        };

        std::vector<cp_t> expected{
            cp_t{0.125681, 0.252712},  cp_t{0.262591, 0.370189},
            cp_t{0.1293, 0.371058},    cp_t{0.348302, 0.183007},
            cp_t{0.303908, 0.0829816}, cp_t{0.18914, 0.179513},
            cp_t{0.173147, 0.0922496}, cp_t{0.298857, 0.269628},
            cp_t{0.125681, 0.252712},  cp_t{0.262591, 0.370189},
            cp_t{0.1293, 0.371058},    cp_t{0.392249, 0.195796},
            cp_t{0.353419, 0.108307},  cp_t{0.18914, 0.179513},
            cp_t{0.173147, 0.0922496}, cp_t{0.298857, 0.269628},
        };

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyDoubleExcitation({0, 1, 2, 3}, false,
                                            {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("DoubleExcitation", {0, 1, 2, 3}, false,
                                     {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyDoubleExcitationMinus",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;

        std::vector<cp_t> ini_st{
            cp_t{0.125681356503, 0.252712197380},
            cp_t{0.262591068130, 0.370189000494},
            cp_t{0.129300299863, 0.371057794075},
            cp_t{0.392248682814, 0.195795523118},
            cp_t{0.303908059240, 0.082981563244},
            cp_t{0.189140284321, 0.179512645957},
            cp_t{0.173146612336, 0.092249594834},
            cp_t{0.298857179897, 0.269627836165},
            cp_t{0.125681356503, 0.252712197380},
            cp_t{0.262591068130, 0.370189000494},
            cp_t{0.129300299863, 0.371057794075},
            cp_t{0.392248682814, 0.195795523118},
            cp_t{0.303908059240, 0.082981563244},
            cp_t{0.189140284321, 0.179512645957},
            cp_t{0.173146612336, 0.092249594834},
            cp_t{0.298857179897, 0.269627836165},
        };

        std::vector<cp_t> expected{
            cp_t{0.158204, 0.233733},  cp_t{0.309533, 0.331939},
            cp_t{0.177544, 0.350543},  cp_t{0.348302, 0.183007},
            cp_t{0.31225, 0.0417871},  cp_t{0.211353, 0.152737},
            cp_t{0.183886, 0.0683795}, cp_t{0.33209, 0.227445},
            cp_t{0.158204, 0.233733},  cp_t{0.309533, 0.331939},
            cp_t{0.177544, 0.350543},  cp_t{0.414822, 0.141837},
            cp_t{0.353419, 0.108307},  cp_t{0.211353, 0.152737},
            cp_t{0.183886, 0.0683795}, cp_t{0.33209, 0.227445},
        };

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyDoubleExcitationMinus({0, 1, 2, 3}, false,
                                                 {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("DoubleExcitationMinus", {0, 1, 2, 3},
                                     false, {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyDoubleExcitationPlus",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;

        std::vector<cp_t> ini_st{
            cp_t{0.125681356503, 0.252712197380},
            cp_t{0.262591068130, 0.370189000494},
            cp_t{0.129300299863, 0.371057794075},
            cp_t{0.392248682814, 0.195795523118},
            cp_t{0.303908059240, 0.082981563244},
            cp_t{0.189140284321, 0.179512645957},
            cp_t{0.173146612336, 0.092249594834},
            cp_t{0.298857179897, 0.269627836165},
            cp_t{0.125681356503, 0.252712197380},
            cp_t{0.262591068130, 0.370189000494},
            cp_t{0.129300299863, 0.371057794075},
            cp_t{0.392248682814, 0.195795523118},
            cp_t{0.303908059240, 0.082981563244},
            cp_t{0.189140284321, 0.179512645957},
            cp_t{0.173146612336, 0.092249594834},
            cp_t{0.298857179897, 0.269627836165},
        };

        std::vector<cp_t> expected{
            cp_t{0.090922, 0.267194},  cp_t{0.210975, 0.40185},
            cp_t{0.0787548, 0.384968}, cp_t{0.348302, 0.183007},
            cp_t{0.290157, 0.122699},  cp_t{0.16356, 0.203093},
            cp_t{0.159325, 0.114478},  cp_t{0.260305, 0.307012},
            cp_t{0.090922, 0.267194},  cp_t{0.210975, 0.40185},
            cp_t{0.0787548, 0.384968}, cp_t{0.362694, 0.246269},
            cp_t{0.353419, 0.108307},  cp_t{0.16356, 0.203093},
            cp_t{0.159325, 0.114478},  cp_t{0.260305, 0.307012},
        };

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyDoubleExcitationPlus({0, 1, 2, 3}, false,
                                                {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<cp_t> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("DoubleExcitationPlus", {0, 1, 2, 3},
                                     false, {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Sample", "[StateVectorKokkosManaged_Param]", float,
                   double) {

    constexpr uint32_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

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

    std::vector<TestType> expected_probabilities = {
        0.67078706, 0.03062806, 0.0870997,  0.00397696,
        0.17564072, 0.00801973, 0.02280642, 0.00104134};

    size_t N = std::pow(2, num_qubits);
    size_t num_samples = 100000;

    auto m = Lightning_Kokkos::Simulators::MeasuresKokkos<TestType>(measure_sv);
    auto samples = m.generate_samples(num_samples);

    std::vector<size_t> counts(N, 0);
    std::vector<size_t> samples_decimal(num_samples, 0);

    // convert samples to decimal and then bin them in counts
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < num_qubits; j++) {
            if (samples[i * num_qubits + j] != 0) {
                samples_decimal[i] += twos[(num_qubits - 1 - j)];
            }
        }
        counts[samples_decimal[i]] += 1;
    }

    // compute estimated probabilities from histogram
    std::vector<TestType> probabilities(counts.size());
    for (size_t i = 0; i < counts.size(); i++) {
        probabilities[i] = counts[i] / (TestType)num_samples;
    }

    // compare estimated probabilities to real probabilities
    SECTION("No wires provided:") {
        REQUIRE_THAT(probabilities,
                     Catch::Approx(expected_probabilities).margin(.05));
    }
}
