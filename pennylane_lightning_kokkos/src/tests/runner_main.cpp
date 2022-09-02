#define CATCH_CONFIG_RUNNER
#include "StateVectorKokkos.hpp"
#include <Kokkos_Core.hpp>
#include <catch2/catch.hpp>

using namespace Pennylane;

int main(int argc, char *argv[]) {
    int result;
    {
        StateVectorKokkos<double> kokkos_sv_fp64{0};
        StateVectorKokkos<float> kokkos_sv_fp32{0};
        result = Catch::Session().run(argc, argv);
    }
    return result;
}
