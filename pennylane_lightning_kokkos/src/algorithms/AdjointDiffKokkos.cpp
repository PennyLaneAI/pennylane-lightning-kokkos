#include "AdjointDiffKokkos.hpp"

// explicit instantiation
template class Pennylane::Algorithms::AdjointJacobianKokkos<float>;
template class Pennylane::Algorithms::AdjointJacobianKokkos<double>;
