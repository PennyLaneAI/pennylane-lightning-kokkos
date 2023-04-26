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

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorPhaseShift",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorPhaseShift({1}, false);
            kokkos_gate_svp.applyPhaseShift({1}, false, {ep});
            kokkos_gate_svm.applyPhaseShift({1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale =
                kokkos_gntr_sv.applyGenerator("PhaseShift", {1}, false);
            kokkos_gate_svp.applyPhaseShift({1}, false, {ep});
            kokkos_gate_svm.applyPhaseShift({1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorIsingXX",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorIsingXX({0, 1}, false);
            kokkos_gate_svp.applyIsingXX({0, 1}, false, {ep});
            kokkos_gate_svm.applyIsingXX({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale =
                kokkos_gntr_sv.applyGenerator("IsingXX", {0, 1}, false);
            kokkos_gate_svp.applyIsingXX({0, 1}, false, {ep});
            kokkos_gate_svm.applyIsingXX({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorIsingXY",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorIsingXY({0, 1}, false);
            kokkos_gate_svp.applyIsingXY({0, 1}, false, {ep});
            kokkos_gate_svm.applyIsingXY({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale =
                kokkos_gntr_sv.applyGenerator("IsingXY", {0, 1}, false);
            kokkos_gate_svp.applyIsingXY({0, 1}, false, {ep});
            kokkos_gate_svm.applyIsingXY({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorIsingYY",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorIsingYY({0, 1}, false);
            kokkos_gate_svp.applyIsingYY({0, 1}, false, {ep});
            kokkos_gate_svm.applyIsingYY({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale =
                kokkos_gntr_sv.applyGenerator("IsingYY", {0, 1}, false);
            kokkos_gate_svp.applyIsingYY({0, 1}, false, {ep});
            kokkos_gate_svm.applyIsingYY({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorIsingZZ",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorIsingZZ({0, 1}, false);
            kokkos_gate_svp.applyIsingZZ({0, 1}, false, {ep});
            kokkos_gate_svm.applyIsingZZ({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale =
                kokkos_gntr_sv.applyGenerator("IsingZZ", {0, 1}, false);
            kokkos_gate_svp.applyIsingZZ({0, 1}, false, {ep});
            kokkos_gate_svm.applyIsingZZ({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "StateVectorKokkosManaged::applyGeneratorControlledPhaseShift",
    "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorControlledPhaseShift(
                {0, 1}, false);
            kokkos_gate_svp.applyControlledPhaseShift({0, 1}, false, {ep});
            kokkos_gate_svm.applyControlledPhaseShift({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("ControlledPhaseShift",
                                                       {0, 1}, false);
            kokkos_gate_svp.applyControlledPhaseShift({0, 1}, false, {ep});
            kokkos_gate_svm.applyControlledPhaseShift({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorCRX",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorCRX({0, 1}, false);
            kokkos_gate_svp.applyCRX({0, 1}, false, {ep});
            kokkos_gate_svm.applyCRX({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("CRX", {0, 1}, false);
            kokkos_gate_svp.applyCRX({0, 1}, false, {ep});
            kokkos_gate_svm.applyCRX({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorCRY",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorCRY({0, 1}, false);
            kokkos_gate_svp.applyCRY({0, 1}, false, {ep});
            kokkos_gate_svm.applyCRY({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("CRY", {0, 1}, false);
            kokkos_gate_svp.applyCRY({0, 1}, false, {ep});
            kokkos_gate_svm.applyCRY({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorCRZ",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorCRZ({0, 1}, false);
            kokkos_gate_svp.applyCRZ({0, 1}, false, {ep});
            kokkos_gate_svm.applyCRZ({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("CRZ", {0, 1}, false);
            kokkos_gate_svp.applyCRZ({0, 1}, false, {ep});
            kokkos_gate_svm.applyCRZ({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorMultiRZ",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorMultiRZ({0}, false);
            kokkos_gate_svp.applyMultiRZ({0}, false, {ep});
            kokkos_gate_svm.applyMultiRZ({0}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("MultiRZ", {0}, false);
            kokkos_gate_svp.applyMultiRZ({0}, false, {ep});
            kokkos_gate_svm.applyMultiRZ({0}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorRX",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorRX({1}, false);
            kokkos_gate_svp.applyRX({1}, false, {ep});
            kokkos_gate_svm.applyRX({1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("RX", {1}, false);
            kokkos_gate_svp.applyRX({1}, false, {ep});
            kokkos_gate_svm.applyRX({1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {

                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorRY",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorRY({1}, false);
            kokkos_gate_svp.applyRY({1}, false, {ep});
            kokkos_gate_svm.applyRY({1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("RY", {1}, false);
            kokkos_gate_svp.applyRY({1}, false, {ep});
            kokkos_gate_svm.applyRY({1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorRZ",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorRZ({1}, false);
            kokkos_gate_svp.applyRZ({1}, false, {ep});
            kokkos_gate_svm.applyRZ({1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("RZ", {1}, false);
            kokkos_gate_svp.applyRZ({1}, false, {ep});
            kokkos_gate_svm.applyRZ({1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorSingleExcitation",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale =
                kokkos_gntr_sv.applyGeneratorSingleExcitation({0, 1}, false);
            kokkos_gate_svp.applySingleExcitation({0, 1}, false, {ep});
            kokkos_gate_svm.applySingleExcitation({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("SingleExcitation",
                                                       {0, 1}, false);
            kokkos_gate_svp.applySingleExcitation({0, 1}, false, {ep});
            kokkos_gate_svm.applySingleExcitation({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "StateVectorKokkosManaged::applyGeneratorSingleExcitationMinus",
    "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorSingleExcitationMinus(
                {0, 1}, false);
            kokkos_gate_svp.applySingleExcitationMinus({0, 1}, false, {ep});
            kokkos_gate_svm.applySingleExcitationMinus({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("SingleExcitationMinus",
                                                       {0, 1}, false);
            kokkos_gate_svp.applySingleExcitationMinus({0, 1}, false, {ep});
            kokkos_gate_svm.applySingleExcitationMinus({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "StateVectorKokkosManaged::applyGeneratorSingleExcitationPlus",
    "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorSingleExcitationPlus(
                {0, 1}, false);
            kokkos_gate_svp.applySingleExcitationPlus({0, 1}, false, {ep});
            kokkos_gate_svm.applySingleExcitationPlus({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("SingleExcitationPlus",
                                                       {0, 1}, false);
            kokkos_gate_svp.applySingleExcitationPlus({0, 1}, false, {ep});
            kokkos_gate_svm.applySingleExcitationPlus({0, 1}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}
TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGeneratorDoubleExcitation",
                   "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorDoubleExcitation(
                {0, 1, 2, 3}, false);
            kokkos_gate_svp.applyDoubleExcitation({0, 1, 2, 3}, false, {ep});
            kokkos_gate_svm.applyDoubleExcitation({0, 1, 2, 3}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("DoubleExcitation",
                                                       {0, 1, 2, 3}, false);
            kokkos_gate_svp.applyDoubleExcitation({0, 1, 2, 3}, false, {ep});
            kokkos_gate_svm.applyDoubleExcitation({0, 1, 2, 3}, false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "StateVectorKokkosManaged::applyGeneratorDoubleExcitationMinus",
    "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorDoubleExcitationMinus(
                {0, 1, 2, 3}, false);
            kokkos_gate_svp.applyDoubleExcitationMinus({0, 1, 2, 3}, false,
                                                       {ep});
            kokkos_gate_svm.applyDoubleExcitationMinus({0, 1, 2, 3}, false,
                                                       {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("DoubleExcitationMinus",
                                                       {0, 1, 2, 3}, false);
            kokkos_gate_svp.applyDoubleExcitationMinus({0, 1, 2, 3}, false,
                                                       {ep});
            kokkos_gate_svm.applyDoubleExcitationMinus({0, 1, 2, 3}, false,
                                                       {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "StateVectorKokkosManaged::applyGeneratorDoubleExcitationPlus",
    "[StateVectorKokkosManaged_Generator]", float, double) {
    {
        using cp_t = Kokkos::complex<TestType>;
        const std::size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGeneratorDoubleExcitationPlus(
                {0, 1, 2, 3}, false);
            kokkos_gate_svp.applyDoubleExcitationPlus({0, 1, 2, 3}, false,
                                                      {ep});
            kokkos_gate_svm.applyDoubleExcitationPlus({0, 1, 2, 3}, false,
                                                      {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<cp_t> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                             {0, 0});
            std::vector<cp_t> result_gate_svp(kokkos_gate_svp.getLength(),
                                              {0, 0});
            std::vector<cp_t> result_gate_svm(kokkos_gate_svm.getLength(),
                                              {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            auto scale = kokkos_gntr_sv.applyGenerator("DoubleExcitationPlus",
                                                       {0, 1, 2, 3}, false);
            kokkos_gate_svp.applyDoubleExcitationPlus({0, 1, 2, 3}, false,
                                                      {ep});
            kokkos_gate_svm.applyDoubleExcitationPlus({0, 1, 2, 3}, false,
                                                      {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (std::size_t j = 0; j < Util::exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}
