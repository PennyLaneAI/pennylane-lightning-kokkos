#include "AdjointDiffKokkos.hpp"

// explicit instantiation
template class Pennylane::Lightning::Kokkos::Algorithms::AdjointJacobianKokkos<
    float>;
template class Pennylane::Lightning::Kokkos::Algorithms::AdjointJacobianKokkos<
    double>;
