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

template <class Precision> struct getProbFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;
    Kokkos::View<Precision *> probilities;

    getProbFunctor(Kokkos::View<Kokkos::complex<Precision> *> arr_,
                   Kokkos::View<Precision *> probilities_)
        : arr(arr_), probilities(probilities_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        Precision REAL = arr[k].real();
        Precision IMAG = arr[k].imag();
        probilities[k] = REAL * REAL + IMAG * IMAG;
    }
};

template <class Precision> struct getCDFFunctor {

    Kokkos::View<Precision *> probilities;

    getCDFFunctor(Kokkos::View<Precision *> probilities_)
        : probilities(probilities_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t &k, Precision &update_value,
                    const bool final) const {
        const Precision val_k = probilities[k];

        if (final)
            probilities[k] = update_value;

        update_value += val_k;
    }
};

template <class Precision, template <class ExecutionSpace> class GeneratorPool,
          class ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct Sampler {

    Kokkos::View<size_t *> samples;
    Kokkos::View<Precision *> cdf;
    GeneratorPool<ExecutionSpace> rand_pool;

    const size_t num_qubits;
    const size_t N;

    Sampler(Kokkos::View<size_t *> samples_, Kokkos::View<Precision *> cdf_,
            GeneratorPool<ExecutionSpace> rand_pool_, const size_t num_qubits_,
            const size_t N_)
        : samples(samples_), cdf(cdf_), rand_pool(rand_pool_),
          num_qubits(num_qubits_), N(N_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool<ExecutionSpace>::generator_type rand_gen =
            rand_pool.get_state();

        Precision U = rand_gen.drand(0.0, 1.0);

        size_t idx;

        if (U <= cdf[1]) {
            idx = 0;
        } else {
            size_t lo = 1, hi = N;
            size_t mid;
            Precision cdf_t;
            while (hi - lo > 1) {
                mid = (hi + lo) / 2;
                if (mid == N)
                    cdf_t = 1;
                else
                    cdf_t = cdf[mid];
                if (cdf_t < U)
                    lo = mid;
                else
                    hi = mid;
            }
            idx = hi - 1;
        }
        for (size_t j = 0; j < num_qubits; j++) {
            samples[k * num_qubits + (num_qubits - 1 - j)] = (idx >> j) & 1U;
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

} // namespace Functors
} // namespace Pennylane
