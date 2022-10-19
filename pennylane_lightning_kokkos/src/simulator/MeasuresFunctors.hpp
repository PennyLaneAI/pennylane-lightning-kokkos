#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "Util.hpp"

namespace {
using namespace Pennylane::Util;
namespace KE = Kokkos::Experimental;
} // namespace

namespace Pennylane {
namespace Functors {

/**
 * @brief Compute probability distribution from StateVector.
 *
 * @param arr_ StateVector data.
 * @param probabilities_ Discrete probability distribution.
 */
template <class Precision> struct getProbFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;
    Kokkos::View<Precision *> probability;

    getProbFunctor(Kokkos::View<Kokkos::complex<Precision> *> arr_,
                   Kokkos::View<Precision *> probability_)
        : arr(arr_), probability(probability_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        Precision REAL = arr[k].real();
        Precision IMAG = arr[k].imag();
        probability[k] = REAL * REAL + IMAG * IMAG;
    }
};

/**
 * @brief Compute sub-probability distribution from StateVector.
 *
 * @param arr_ StateVector data.
 * @param probability_ Sub discrete probability distribution.
 */
template <class Precision> struct getSubProbFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;
    Kokkos::View<Precision *> probability;
    Kokkos::View<size_t *> all_indices;
    Kokkos::View<size_t *> all_offsets;

    getSubProbFunctor(Kokkos::View<Kokkos::complex<Precision> *> arr_,
                      Kokkos::View<Precision *> probability_,
                      Kokkos::View<size_t *> all_indices_,
                      Kokkos::View<size_t *> all_offsets_)
        : arr(arr_), probability(probability_), all_indices(all_indices_),
          all_offsets(all_offsets_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i, const size_t j) const {
        size_t index = all_indices[i] + all_offsets[j];
        Precision REAL = arr[index].real();
        Precision IMAG = arr[index].imag();
        Precision value = REAL * REAL + IMAG * IMAG;
        Kokkos::atomic_add(&probability[i], value);
    }
};

template <class Precision> struct getTransposedIndexFunctor {

    Kokkos::View<size_t *> sorted_ind_wires;
    Kokkos::View<size_t *> trans_index;
    getTransposedIndexFunctor(Kokkos::View<size_t *> sorted_ind_wires_,
                              Kokkos::View<size_t *> trans_index_)
        : sorted_ind_wires(sorted_ind_wires_), trans_index(trans_index_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i, const size_t j) const {

        size_t axis = sorted_ind_wires[j];
        size_t index = i / (1L << j);
        size_t sub_index = (index % 2) << axis;

        Kokkos::atomic_add(&trans_index[i], sub_index);
    }
};

template <class Precision> struct getTransposedFunctor {

    Kokkos::View<Precision *> transProb;
    Kokkos::View<Precision *> probability;
    Kokkos::View<size_t *> trans_index;
    getTransposedFunctor(Kokkos::View<Precision *> transProb_,
                         Kokkos::View<Precision *> probability_,
                         Kokkos::View<size_t *> trans_index_)
        : transProb(transProb_), probability(probability_),
          trans_index(trans_index_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const {

        size_t new_index = trans_index[i];
        transProb[new_index] = probability[i];
    }
};

} // namespace Functors
} // namespace Pennylane
