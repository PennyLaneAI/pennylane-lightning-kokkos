#include "AdjointDiffKokkos.hpp"

// explicit instantiation
template class Pennylane::LKokkos::Algorithms::AdjointJacobianKokkos<float>;
template class Pennylane::LKokkos::Algorithms::AdjointJacobianKokkos<double>;
