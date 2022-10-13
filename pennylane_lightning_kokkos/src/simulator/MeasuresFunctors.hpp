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

template <class Precision> struct getLocalCDFFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;
    Kokkos::View<Precision *> cdf;
    size_t Nsqrt;
    size_t N;

    getLocalCDFFunctor(Kokkos::View<Kokkos::complex<Precision> *> arr_,
                       Kokkos::View<Precision *> cdf_, const size_t Nsqrt_,
                       const size_t N_) {
        arr = arr_;
        cdf = cdf_;
        Nsqrt = Nsqrt_;
        N = N_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {

        for (size_t i = 0; i < Nsqrt; i++) {
            size_t idx = i + k * Nsqrt;
	    Precision REAL = arr[idx].real();
	    Precision IMAG = arr[idx].imag();
	    Precision NORM = REAL*REAL + IMAG*IMAG;
            if (i == 0)
                cdf[idx] = NORM;
            else if (i < N)
                cdf[idx] = NORM + cdf[idx - 1];
        }
    }
};

template <class Precision> struct getGlobalCDFFunctor {

    Kokkos::View<Precision *> cdf;
    size_t i;
    size_t Nsqrt;
    size_t N;

    getGlobalCDFFunctor(Kokkos::View<Precision *> cdf_, size_t i_,
                        const size_t Nsqrt_, const size_t N_) {
        cdf = cdf_;
        i = i_;
        Nsqrt = Nsqrt_;
        N = N_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {

        size_t idx = i * Nsqrt + k;
        size_t idx0 = i * Nsqrt - 1;

        if (idx < N)
            cdf[idx] += cdf[idx0];
    }
};

template <class Precision, template <class ExecutionSpace> class GeneratorPool,
          class ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct Sampler {

    Kokkos::View<size_t *> samples;

    Kokkos::View<Precision *> cdf;

    GeneratorPool<ExecutionSpace> rand_pool;

    size_t num_qubits;

    size_t N;

    Sampler(Kokkos::View<size_t *> samples_, Kokkos::View<Precision *> cdf_,
            GeneratorPool<ExecutionSpace> rand_pool_, const size_t num_qubits_,
            const size_t N_) {
        samples = samples_;
        cdf = cdf_;
        rand_pool = rand_pool_;
        num_qubits = num_qubits_;
        N = N_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool<ExecutionSpace>::generator_type rand_gen =
            rand_pool.get_state();

        Precision U = rand_gen.drand(0.0, 1.0);

        size_t idx;

        if (U <= cdf[0]) {
            idx = 0;
        } else {
            size_t lo = 0, hi = N - 1;
            size_t mid;
            while (hi - lo > 1) {
                mid = (hi + lo) / 2;
                if (cdf[mid] < U)
                    lo = mid;
                else
                    hi = mid;
            }
            idx = hi;
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
