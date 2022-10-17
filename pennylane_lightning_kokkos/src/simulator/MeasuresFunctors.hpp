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
 *@brief Compute cumulative probability distribution from discrete probability
 *distribution and the cumulative probability distribution is then stored into
 *probabilities_.
 *
 *@param probabilities_ Discrete probability distribution.
 */
template <class Precision> struct getCDFFunctor {

    Kokkos::View<Precision *> probability;

    getCDFFunctor(Kokkos::View<Precision *> probability_)
        : probability(probability_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t &k, Precision &update_value,
                    const bool fin) const {
        const Precision val_k = probability[k];

        if (fin)
            probability[k] = update_value;

        update_value += val_k;
    }
};

/**
 *@brief Sampling using Random_XorShift64_Pool
 *
 * @param samples_ Kokkos::View of the generated samples.
 * @param cdf_  Kokkos::View of cumulative probability distribution.
 * @param rand_pool_ The generatorPool.
 * @param num_qubits_ Number of qubits.
 * @param length_ Length of cumulative probability distribution.
 */

template <class Precision, template <class ExecutionSpace> class GeneratorPool,
          class ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct Sampler {

    Kokkos::View<size_t *> samples;
    Kokkos::View<Precision *> cdf;
    GeneratorPool<ExecutionSpace> rand_pool;

    const size_t num_qubits;
    const size_t length;

    Sampler(Kokkos::View<size_t *> samples_, Kokkos::View<Precision *> cdf_,
            GeneratorPool<ExecutionSpace> rand_pool_, const size_t num_qubits_,
            const size_t length_)
        : samples(samples_), cdf(cdf_), rand_pool(rand_pool_),
          num_qubits(num_qubits_), length(length_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        // Get a random number state from the pool for the active thread
        auto rand_gen = rand_pool.get_state();
        Precision U_rand = rand_gen.drand(0.0, 1.0);
        size_t index;

        // Binary search for the bin index of cumulative probability
        // distribution that generated random number U falls into.
        if (U_rand <= cdf[1]) {
            index = 0;
        } else {
            size_t low_idx = 1, high_idx = length;
            size_t mid_idx;
            Precision cdf_t;
            while (high_idx - low_idx > 1) {
                mid_idx = (high_idx + low_idx) / 2;
                if (mid_idx == length)
                    cdf_t = 1;
                else
                    cdf_t = cdf[mid_idx];
                if (cdf_t < U_rand)
                    low_idx = mid_idx;
                else
                    high_idx = mid_idx;
            }
            index = high_idx - 1;
        }
        for (size_t j = 0; j < num_qubits; j++) {
            samples[k * num_qubits + (num_qubits - 1 - j)] = (index >> j) & 1U;
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

} // namespace Functors
} // namespace Pennylane
