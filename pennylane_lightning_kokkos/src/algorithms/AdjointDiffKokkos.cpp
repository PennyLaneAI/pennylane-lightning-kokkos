#include "AdjointDiffKokkos.hpp"

// explicit instantiation
template class Pennylane::Lightning_Kokkos::Algorithms::AdjointJacobianKokkos<
    float>;
template class Pennylane::Lightning_Kokkos::Algorithms::AdjointJacobianKokkos<
    double>;
