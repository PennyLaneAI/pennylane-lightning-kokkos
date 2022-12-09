#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // for C++
#endif
#include <cmath>

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "Util.hpp"

namespace {
using namespace Pennylane::Util;
namespace KE = Kokkos::Experimental;
} // namespace

namespace Pennylane {
namespace Functors {

template <class Precision, bool inverse = false>
struct applySingleQubitOpFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;
    Kokkos::View<Kokkos::complex<Precision> *> matrix;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    applySingleQubitOpFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits,
        const Kokkos::View<Kokkos::complex<Precision> *> &matrix_,
        const std::vector<size_t> &wires) {
        arr = arr_;
        matrix = matrix_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        if constexpr (inverse) {
            const std::size_t i0 =
                ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const std::size_t i1 = i0 | rev_wire_shift;
            const Kokkos::complex<Precision> v0 = arr[i0];
            const Kokkos::complex<Precision> v1 = arr[i1];

            arr[i0] =
                conj(matrix[0B00]) * v0 +
                conj(matrix[0B10]) * v1; // NOLINT(readability-magic-numbers)
            arr[i1] =
                conj(matrix[0B01]) * v0 +
                conj(matrix[0B11]) * v1; // NOLINT(readability-magic-numbers)
                                         // }
        } else {
            const std::size_t i0 =
                ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const std::size_t i1 = i0 | rev_wire_shift;
            const Kokkos::complex<Precision> v0 = arr[i0];
            const Kokkos::complex<Precision> v1 = arr[i1];
            arr[i0] = matrix[0B00] * v0 +
                      matrix[0B01] * v1; // NOLINT(readability-magic-numbers)
            arr[i1] = matrix[0B10] * v0 +
                      matrix[0B11] * v1; // NOLINT(readability-magic-numbers)
        }
    }
};

/**
 * @brief Apply a two qubit gate to the statevector.
 *
 * @param arr Pointer to the statevector.
 * @param num_qubits Number of qubits.
 * @param matrix Perfect square matrix in row-major order.
 * @param wires Wires the gate applies to.
 * @param inverse Indicate whether inverse should be taken.
 */
template <class Precision, bool inverse = false> struct applyTwoQubitOpFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;
    Kokkos::View<Kokkos::complex<Precision> *> matrix;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyTwoQubitOpFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits,
        const Kokkos::View<Kokkos::complex<Precision> *> &matrix_,
        const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
        matrix = matrix_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        if constexpr (inverse) {
            const std::size_t i00 = ((k << 2U) & parity_high) |
                                    ((k << 1U) & parity_middle) |
                                    (k & parity_low);
            const std::size_t i10 = i00 | rev_wire1_shift;
            const std::size_t i01 = i00 | rev_wire0_shift;
            const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const Kokkos::complex<Precision> v00 = arr[i00];
            const Kokkos::complex<Precision> v01 = arr[i01];
            const Kokkos::complex<Precision> v10 = arr[i10];
            const Kokkos::complex<Precision> v11 = arr[i11];

            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i00] = conj(matrix[0b0000]) * v00 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b0100]) * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1000]) * v10 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1100]) * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i01] = conj(matrix[0b0001]) * v00 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b0101]) * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1001]) * v10 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1101]) * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i10] = conj(matrix[0b0010]) * v00 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b0110]) * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1010]) * v10 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1110]) * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i11] = conj(matrix[0b0011]) * v00 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b0111]) * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1011]) * v10 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1111]) * v11;
        } else {
            const std::size_t i00 = ((k << 2U) & parity_high) |
                                    ((k << 1U) & parity_middle) |
                                    (k & parity_low);
            const std::size_t i10 = i00 | rev_wire1_shift;
            const std::size_t i01 = i00 | rev_wire0_shift;
            const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const Kokkos::complex<Precision> v00 = arr[i00];
            const Kokkos::complex<Precision> v01 = arr[i01];
            const Kokkos::complex<Precision> v10 = arr[i10];
            const Kokkos::complex<Precision> v11 = arr[i11];

            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i00] = matrix[0b0000] * v00 + matrix[0b0001] * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       matrix[0b0010] * v10 + matrix[0b0011] * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i01] = matrix[0b0100] * v00 + matrix[0b0101] * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       matrix[0b0110] * v10 + matrix[0b0111] * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i10] = matrix[0b1000] * v00 + matrix[0b1001] * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       matrix[0b1010] * v10 + matrix[0b1011] * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i11] = matrix[0b1100] * v00 + matrix[0b1101] * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       matrix[0b1110] * v10 + matrix[0b1111] * v11;
            // }
        }
    }
};

template <class Precision, bool inverse = false>
struct applyMultiQubitOpFunctor {

    using KokkosComplexVector = Kokkos::View<Kokkos::complex<Precision> *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector indices;
    KokkosIntVector wires;
    KokkosComplexVector coeffs_in;
    std::size_t dim;
    std::size_t num_qubits;

    applyMultiQubitOpFunctor(KokkosComplexVector &arr_, std::size_t num_qubits_,
                             const KokkosComplexVector &matrix_,
                             KokkosIntVector &wires_) {
        dim = 1U << wires_.size();
        indices = KokkosIntVector("indices", dim);
        coeffs_in = KokkosComplexVector("coeffs_in", dim);
        num_qubits = num_qubits_;
        wires = wires_;
        arr = arr_;
        matrix = matrix_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(std::size_t kp) const {
        const std::size_t k = kp * dim;
        using Pennylane::Util::bitswap;
        if constexpr (inverse) {

            for (size_t inner_idx = 0; inner_idx < dim; inner_idx++) {
                std::size_t idx = k | inner_idx;
                const std::size_t n_wires = dim;

                for (std::size_t pos = 0; pos < n_wires; pos++) {
                    size_t x = ((idx >> (n_wires - pos - 1)) ^
                                (idx >> (num_qubits - wires[pos] - 1))) &
                               1U;
                    idx = idx ^ ((x << (n_wires - pos - 1)) |
                                 (x << (num_qubits - wires[pos] - 1)));
                }

                indices[inner_idx] = idx;
                coeffs_in[inner_idx] = arr[idx];
            }

            for (size_t i = 0; i < dim; i++) {
                const auto idx = indices[i];
                arr[idx] = 0.0;

                for (size_t j = 0; j < dim; j++) {
                    const std::size_t base_idx = j * dim;
                    arr[idx] +=
                        Kokkos::conj(matrix[base_idx + i]) * coeffs_in[j];
                }
            }
        } else {
            for (size_t inner_idx = 0; inner_idx < dim; inner_idx++) {
                std::size_t idx = k | inner_idx;
                const std::size_t n_wires = wires.size();

                for (std::size_t pos = 0; pos < n_wires; pos++) {
                    size_t x = ((idx >> (n_wires - pos - 1)) ^
                                (idx >> (num_qubits - wires[pos] - 1))) &
                               1U;
                    idx = idx ^ ((x << (n_wires - pos - 1)) |
                                 (x << (num_qubits - wires[pos] - 1)));
                }

                indices[inner_idx] = idx;
                coeffs_in[inner_idx] = arr[idx];
            }

            for (size_t i = 0; i < dim; i++) {
                const auto idx = indices[i];
                arr[idx] = 0.0;
                const std::size_t base_idx = i * dim;

                for (size_t j = 0; j < dim; j++) {
                    arr[idx] += matrix[base_idx + j] * coeffs_in[j];
                }
            }
        }
    }
};

template <class Precision, bool inverse = false> struct applyHadamardFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    applyHadamardFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                         std::size_t num_qubits,
                         const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        if constexpr (inverse) {
            const std::size_t i0 =
                ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const std::size_t i1 = i0 | rev_wire_shift;
            const Kokkos::complex<Precision> v0 = arr[i0];
            const Kokkos::complex<Precision> v1 = arr[i1];
            arr[i0] = M_SQRT1_2 * v0 +
                      M_SQRT1_2 * v1; // NOLINT(readability-magic-numbers)
            arr[i1] = M_SQRT1_2 * v0 +
                      (-M_SQRT1_2) * v1; // NOLINT(readability-magic-numbers)
                                         // }
        } else {
            const std::size_t i0 =
                ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const std::size_t i1 = i0 | rev_wire_shift;
            const Kokkos::complex<Precision> v0 = arr[i0];
            const Kokkos::complex<Precision> v1 = arr[i1];
            arr[i0] = M_SQRT1_2 * v0 +
                      M_SQRT1_2 * v1; // NOLINT(readability-magic-numbers)
            arr[i1] = M_SQRT1_2 * v0 +
                      -M_SQRT1_2 * v1; // NOLINT(readability-magic-numbers)
                                       // }
        }
    }
};

template <class Precision, bool inverse = false> struct applyPauliXFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    applyPauliXFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                       std::size_t num_qubits,
                       const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        KE::swap(arr[i0], arr[i1]);
    }
};

template <class Precision, bool inverse = false> struct applyPauliYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    applyPauliYFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                       std::size_t num_qubits,
                       const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr[i0];
        const auto v1 = arr[i1];
        arr[i0] = Kokkos::complex<Precision>{imag(v1), -real(v1)};
        arr[i1] = Kokkos::complex<Precision>{-imag(v0), real(v0)};
    }
};

template <class Precision, bool inverse = false> struct applyPauliZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    applyPauliZFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                       std::size_t num_qubits,
                       const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        arr[i1] *= -1;
    }
};

template <class Precision, bool inverse = false> struct applySFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;
    Kokkos::complex<Precision> shift;

    applySFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                  std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        shift = (inverse) ? -Kokkos::complex(0, 1) : Kokkos::complex(0, 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        arr[i1] *= shift;
    }
};

template <class Precision, bool inverse = false> struct applyTFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;
    Kokkos::complex<Precision> shift;

    applyTFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                  std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        shift = (inverse) ? conj(exp(Kokkos::complex<Precision>(
                                0, static_cast<Precision>(M_PI / 4))))
                          : exp(Kokkos::complex<Precision>(
                                0, static_cast<Precision>(M_PI / 4)));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        arr[i1] *= shift;
    }
};

template <class Precision, bool inverse = false> struct applyPhaseShiftFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;
    Kokkos::complex<Precision> s;

    applyPhaseShiftFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                           std::size_t num_qubits,
                           const std::vector<size_t> &wires,
                           const Precision &angle) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        s = inverse ? exp(-Kokkos::complex<Precision>(0, angle))
                    : exp(Kokkos::complex<Precision>(0, angle));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        arr[i1] *= s;
    }
};

template <class Precision, bool inverse = false> struct applyRXFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;
    Precision c;
    Precision s;
    applyRXFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                   std::size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<Precision> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const Precision &angle = params[0];
        c = cos(angle * static_cast<Precision>(0.5));
        s = (inverse) ? sin(angle * static_cast<Precision>(0.5))
                      : sin(-angle * static_cast<Precision>(0.5));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr[i0];
        const auto v1 = arr[i1];

        arr[i0] =
            c * v0 + Kokkos::complex<Precision>{-imag(v1) * s, real(v1) * s};
        arr[i1] =
            Kokkos::complex<Precision>{-imag(v0) * s, real(v0) * s} + c * v1;
    }
};

template <class Precision, bool inverse = false> struct applyRYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;
    Precision c;
    Precision s;
    applyRYFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                   std::size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<Precision> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const Precision &angle = params[0];
        c = cos(angle * static_cast<Precision>(0.5));
        s = (inverse) ? -sin(angle * static_cast<Precision>(0.5))
                      : sin(angle * static_cast<Precision>(0.5));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr[i0];
        const auto v1 = arr[i1];

        arr[i0] = Kokkos::complex<Precision>{c * real(v0) - s * real(v1),
                                             c * imag(v0) - s * imag(v1)};
        arr[i1] = Kokkos::complex<Precision>{s * real(v0) + c * real(v1),
                                             s * imag(v0) + c * imag(v1)};
    }
};

template <class Precision, bool inverse = false> struct applyRZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;
    Kokkos::complex<Precision> shift_0;
    Kokkos::complex<Precision> shift_1;

    applyRZFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                   std::size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<Precision> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const Precision &angle = params[0];
        Precision cos_angle = cos(angle * static_cast<Precision>(0.5));
        Precision sin_angle = sin(angle * static_cast<Precision>(0.5));
        Kokkos::complex<Precision> first{cos_angle, -sin_angle};
        Kokkos::complex<Precision> second{cos_angle, sin_angle};
        shift_0 = (inverse) ? Kokkos::conj(first) : first;
        shift_1 = (inverse) ? Kokkos::conj(second) : second;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        arr[i0] *= shift_0;
        arr[i1] *= shift_1;
    }
};

template <class Precision, bool inverse = false> struct applyCNOTFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyCNOTFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                     std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;

        KE::swap(arr[i10], arr[i11]);
    }
};

template <class Precision, bool inverse = false> struct applyCYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyCYFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                   std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;
        Kokkos::complex<Precision> v10 = arr[i10];
        arr[i10] = Kokkos::complex<Precision>{imag(arr[i11]), -real(arr[i11])};
        arr[i11] = Kokkos::complex<Precision>{-imag(v10), real(v10)};
    }
};

template <class Precision, bool inverse = false> struct applyCZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyCZFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                   std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;
        arr[i11] *= -1;
    }
};
template <class Precision, bool inverse = false> struct applyCRotFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Kokkos::complex<Precision> rot_mat_0b00;
    Kokkos::complex<Precision> rot_mat_0b10;
    Kokkos::complex<Precision> rot_mat_0b01;
    Kokkos::complex<Precision> rot_mat_0b11;

    applyCRotFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                     std::size_t num_qubits, const std::vector<size_t> &wires,
                     const std::vector<Precision> &params) {

        const Precision phi = (inverse) ? -params[2] : params[0];
        const Precision theta = (inverse) ? -params[1] : params[1];
        const Precision omega = (inverse) ? -params[0] : params[2];
        const Precision c = std::cos(theta / 2);
        const Precision s = std::sin(theta / 2);
        const Precision p{phi + omega};
        const Precision m{phi - omega};

        auto imag = Kokkos::complex<Precision>(0, 1);
        rot_mat_0b00 = Kokkos::exp(static_cast<Precision>(p / 2) * (-imag)) * c;
        rot_mat_0b01 = -Kokkos::exp(static_cast<Precision>(m / 2) * imag) * s;
        rot_mat_0b10 = Kokkos::exp(static_cast<Precision>(m / 2) * (-imag)) * s;
        rot_mat_0b11 = Kokkos::exp(static_cast<Precision>(p / 2) * imag) * c;

        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const auto v0 = arr[i10];
        const auto v1 = arr[i11];

        arr[i10] = rot_mat_0b00 * v0 + rot_mat_0b01 * v1;
        arr[i11] = rot_mat_0b10 * v0 + rot_mat_0b11 * v1;
    }
};

template <class Precision, bool inverse = false> struct applySWAPFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applySWAPFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                     std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        KE::swap(arr[i10], arr[i01]);
    }
};

template <class Precision, bool inverse = false> struct applyIsingXXFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Precision cr;
    Precision sj;

    applyIsingXXFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                        std::size_t num_qubits,
                        const std::vector<size_t> &wires,
                        const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const Precision &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<Precision> v00 = arr[i00];
        const Kokkos::complex<Precision> v01 = arr[i01];
        const Kokkos::complex<Precision> v10 = arr[i10];
        const Kokkos::complex<Precision> v11 = arr[i11];

        arr[i00] = Kokkos::complex<Precision>{cr * real(v00) + sj * imag(v11),
                                              cr * imag(v00) - sj * real(v11)};
        arr[i01] = Kokkos::complex<Precision>{cr * real(v01) + sj * imag(v10),
                                              cr * imag(v01) - sj * real(v10)};
        arr[i10] = Kokkos::complex<Precision>{cr * real(v10) + sj * imag(v01),
                                              cr * imag(v10) - sj * real(v01)};
        arr[i11] = Kokkos::complex<Precision>{cr * real(v11) + sj * imag(v00),
                                              cr * imag(v11) - sj * real(v00)};
    }
};

template <class Precision, bool inverse = false> struct applyIsingXYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Precision cr;
    Precision sj;

    applyIsingXYFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                        std::size_t num_qubits,
                        const std::vector<size_t> &wires,
                        const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const Precision &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<Precision> v00 = arr[i00];
        const Kokkos::complex<Precision> v01 = arr[i01];
        const Kokkos::complex<Precision> v10 = arr[i10];
        const Kokkos::complex<Precision> v11 = arr[i11];

        arr[i00] = Kokkos::complex<Precision>{real(v00), imag(v00)};
        arr[i01] = Kokkos::complex<Precision>{cr * real(v01) - sj * imag(v10),
                                              cr * imag(v01) + sj * real(v10)};
        arr[i10] = Kokkos::complex<Precision>{cr * real(v10) - sj * imag(v01),
                                              cr * imag(v10) + sj * real(v01)};
        arr[i11] = Kokkos::complex<Precision>{real(v11), imag(v11)};
    }
};

template <class Precision, bool inverse = false> struct applyIsingYYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Precision cr;
    Precision sj;

    applyIsingYYFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                        std::size_t num_qubits,
                        const std::vector<size_t> &wires,
                        const std::vector<Precision> &params) {
        const Precision &angle = params[0];
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<Precision> v00 = arr[i00];
        const Kokkos::complex<Precision> v01 = arr[i01];
        const Kokkos::complex<Precision> v10 = arr[i10];
        const Kokkos::complex<Precision> v11 = arr[i11];

        arr[i00] = Kokkos::complex<Precision>{cr * real(v00) - sj * imag(v11),
                                              cr * imag(v00) + sj * real(v11)};
        arr[i01] = Kokkos::complex<Precision>{cr * real(v01) + sj * imag(v10),
                                              cr * imag(v01) - sj * real(v10)};
        arr[i10] = Kokkos::complex<Precision>{cr * real(v10) + sj * imag(v01),
                                              cr * imag(v10) - sj * real(v01)};
        arr[i11] = Kokkos::complex<Precision>{cr * real(v11) - sj * imag(v00),
                                              cr * imag(v11) + sj * real(v00)};
    }
};

template <class Precision, bool inverse = false> struct applyIsingZZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Kokkos::complex<Precision> first;
    Kokkos::complex<Precision> second;
    Kokkos::complex<Precision> shift_0;
    Kokkos::complex<Precision> shift_1;

    applyIsingZZFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                        std::size_t num_qubits,
                        const std::vector<size_t> &wires,
                        const std::vector<Precision> &params) {
        const Precision &angle = params[0];

        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        first = Kokkos::complex<Precision>{std::cos(angle / 2),
                                           -std::sin(angle / 2)};
        second = Kokkos::complex<Precision>{std::cos(angle / 2),
                                            std::sin(angle / 2)};

        shift_0 = (inverse) ? conj(first) : first;
        shift_1 = (inverse) ? conj(second) : second;

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] *= shift_0;
        arr[i01] *= shift_1;
        arr[i10] *= shift_1;
        arr[i11] *= shift_0;
    }
};
template <class Precision, bool inverse = false>
struct applySingleExcitationFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Precision cr;
    Precision sj;

    applySingleExcitationFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const Precision &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;

        const Kokkos::complex<Precision> v01 = arr[i01];
        const Kokkos::complex<Precision> v10 = arr[i10];

        arr[i01] = cr * v01 - sj * v10;
        arr[i10] = sj * v01 + cr * v10;
    }
};

template <class Precision, bool inverse = false>
struct applySingleExcitationMinusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Precision cr;
    Precision sj;
    Kokkos::complex<Precision> e;

    applySingleExcitationMinusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const Precision &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<Precision>(0, angle / 2))
                    : exp(Kokkos::complex<Precision>(0, -angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<Precision> v01 = arr[i01];
        const Kokkos::complex<Precision> v10 = arr[i10];

        arr[i00] *= e;
        arr[i01] = cr * v01 - sj * v10;
        arr[i10] = sj * v01 + cr * v10;
        arr[i11] *= e;
    }
};

template <class Precision, bool inverse = false>
struct applySingleExcitationPlusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Precision cr;
    Precision sj;
    Kokkos::complex<Precision> e;

    applySingleExcitationPlusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const Precision &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<Precision>(0, -angle / 2))
                    : exp(Kokkos::complex<Precision>(0, angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<Precision> v01 = arr[i01];
        const Kokkos::complex<Precision> v10 = arr[i10];

        arr[i00] *= e;
        arr[i01] = cr * v01 - sj * v10;
        arr[i10] = sj * v01 + cr * v10;
        arr[i11] *= e;
    }
};

template <class Precision, bool inverse = false>
struct applyDoubleExcitationFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire3;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire3_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_min_mid;
    std::size_t rev_wire_max_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;
    Kokkos::complex<Precision> shifts_2;
    Kokkos::complex<Precision> shifts_3;

    Precision cr;
    Precision sj;

    applyDoubleExcitationFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        const std::vector<Precision> &params) {

        const Precision &angle = params[0];
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0000 =
            ((k << 4U) & parity_high) | ((k << 3U) & parity_hmiddle) |
            ((k << 2U) & parity_middle) | ((k << 1U) & parity_lmiddle) |
            (k & parity_low);
        const std::size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;

        const Kokkos::complex<Precision> v3 = arr[i0011];
        const Kokkos::complex<Precision> v12 = arr[i1100];

        arr[i0011] = cr * v3 - sj * v12;
        arr[i1100] = sj * v3 + cr * v12;
    }
};

template <class Precision, bool inverse = false>
struct applyDoubleExcitationMinusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire3;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire3_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_min_mid;
    std::size_t rev_wire_max_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;
    Kokkos::complex<Precision> shifts_2;
    Kokkos::complex<Precision> shifts_3;

    Precision cr;
    Precision sj;
    Kokkos::complex<Precision> e;

    applyDoubleExcitationMinusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        const std::vector<Precision> &params) {
        const Precision &angle = params[0];
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<Precision>(0, angle / 2))
                    : exp(Kokkos::complex<Precision>(0, -angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0000 =
            ((k << 4U) & parity_high) | ((k << 3U) & parity_hmiddle) |
            ((k << 2U) & parity_middle) | ((k << 1U) & parity_lmiddle) |
            (k & parity_low);
        const std::size_t i0001 = i0000 | rev_wire0_shift;
        const std::size_t i0010 = i0000 | rev_wire1_shift;
        const std::size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i0100 = i0000 | rev_wire2_shift;
        const std::size_t i0101 = i0000 | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i0110 = i0000 | rev_wire2_shift | rev_wire1_shift;
        const std::size_t i0111 =
            i0000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1000 = i0000 | rev_wire3_shift;
        const std::size_t i1001 = i0000 | rev_wire3_shift | rev_wire0_shift;
        const std::size_t i1010 = i0000 | rev_wire3_shift | rev_wire1_shift;
        const std::size_t i1011 =
            i0000 | rev_wire3_shift | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;
        const std::size_t i1101 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i1110 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire1_shift;
        const std::size_t i1111 = i0000 | rev_wire3_shift | rev_wire2_shift |
                                  rev_wire1_shift | rev_wire0_shift;

        const Kokkos::complex<Precision> v3 = arr[i0011];
        const Kokkos::complex<Precision> v12 = arr[i1100];

        arr[i0000] *= e;
        arr[i0001] *= e;
        arr[i0010] *= e;
        arr[i0011] = cr * v3 - sj * v12;
        arr[i0100] *= e;
        arr[i0101] *= e;
        arr[i0110] *= e;
        arr[i0111] *= e;
        arr[i1000] *= e;
        arr[i1001] *= e;
        arr[i1010] *= e;
        arr[i1011] *= e;
        arr[i1100] = sj * v3 + cr * v12;
        arr[i1101] *= e;
        arr[i1110] *= e;
        arr[i1111] *= e;
    }
};

template <class Precision, bool inverse = false>
struct applyDoubleExcitationPlusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire3;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire3_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_min_mid;
    std::size_t rev_wire_max_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;
    Kokkos::complex<Precision> shifts_2;
    Kokkos::complex<Precision> shifts_3;

    Precision cr;
    Precision sj;
    Kokkos::complex<Precision> e;

    applyDoubleExcitationPlusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        const std::vector<Precision> &params) {
        const Precision &angle = params[0];
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<Precision>(0, -angle / 2))
                    : exp(Kokkos::complex<Precision>(0, angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {

        const std::size_t i0000 =
            ((k << 4U) & parity_high) | ((k << 3U) & parity_hmiddle) |
            ((k << 2U) & parity_middle) | ((k << 1U) & parity_lmiddle) |
            (k & parity_low);
        const std::size_t i0001 = i0000 | rev_wire0_shift;
        const std::size_t i0010 = i0000 | rev_wire1_shift;
        const std::size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i0100 = i0000 | rev_wire2_shift;
        const std::size_t i0101 = i0000 | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i0110 = i0000 | rev_wire2_shift | rev_wire1_shift;
        const std::size_t i0111 =
            i0000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1000 = i0000 | rev_wire3_shift;
        const std::size_t i1001 = i0000 | rev_wire3_shift | rev_wire0_shift;
        const std::size_t i1010 = i0000 | rev_wire3_shift | rev_wire1_shift;
        const std::size_t i1011 =
            i0000 | rev_wire3_shift | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;
        const std::size_t i1101 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i1110 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire1_shift;
        const std::size_t i1111 = i0000 | rev_wire3_shift | rev_wire2_shift |
                                  rev_wire1_shift | rev_wire0_shift;

        const Kokkos::complex<Precision> v3 = arr[i0011];
        const Kokkos::complex<Precision> v12 = arr[i1100];

        arr[i0000] *= e;
        arr[i0001] *= e;
        arr[i0010] *= e;
        arr[i0011] = cr * v3 - sj * v12;
        arr[i0100] *= e;
        arr[i0101] *= e;
        arr[i0110] *= e;
        arr[i0111] *= e;
        arr[i1000] *= e;
        arr[i1001] *= e;
        arr[i1010] *= e;
        arr[i1011] *= e;
        arr[i1100] = sj * v3 + cr * v12;
        arr[i1101] *= e;
        arr[i1110] *= e;
        arr[i1111] *= e;
    }
};

template <class Precision, bool inverse = false>
struct applyControlledPhaseShiftFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Kokkos::complex<Precision> s;

    applyControlledPhaseShiftFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        const std::vector<Precision> &params) {
        const Precision &angle = params[0];
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        s = inverse ? exp(-Kokkos::complex<Precision>(0, angle))
                    : exp(Kokkos::complex<Precision>(0, angle));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;

        arr[i11] *= s;
    }
};

template <class Precision, bool inverse = false> struct applyCRXFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Precision c;
    Precision js;

    applyCRXFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                    std::size_t num_qubits, const std::vector<size_t> &wires,
                    const std::vector<Precision> &params) {
        const Precision &angle = params[0];
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        c = std::cos(angle / 2);
        js = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<Precision> v10 = arr[i10];
        const Kokkos::complex<Precision> v11 = arr[i11];

        arr[i10] = Kokkos::complex<Precision>{
            c * Kokkos::real(v10) + js * Kokkos::imag(v11),
            c * Kokkos::imag(v10) - js * Kokkos::real(v11)};
        arr[i11] = Kokkos::complex<Precision>{
            c * Kokkos::real(v11) + js * Kokkos::imag(v10),
            c * Kokkos::imag(v11) - js * Kokkos::real(v10)};
    }
};

template <class Precision, bool inverse = false> struct applyCRYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Precision c;
    Precision s;

    applyCRYFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                    std::size_t num_qubits, const std::vector<size_t> &wires,
                    const std::vector<Precision> &params) {
        const Precision &angle = params[0];
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        c = std::cos(angle / 2);
        s = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<Precision> v10 = arr[i10];
        const Kokkos::complex<Precision> v11 = arr[i11];

        arr[i10] = c * v10 - s * v11;
        arr[i11] = s * v10 + c * v11;
    }
};

template <class Precision, bool inverse = false> struct applyCRZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;

    applyCRZFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                    std::size_t num_qubits, const std::vector<size_t> &wires,
                    const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const Precision &angle = params[0];

        const Kokkos::complex<Precision> first = Kokkos::complex<Precision>{
            std::cos(angle / 2), -std::sin(angle / 2)};
        const Kokkos::complex<Precision> second = Kokkos::complex<Precision>{
            std::cos(angle / 2), std::sin(angle / 2)};

        shifts_0 = (inverse) ? conj(first) : first;
        shifts_1 = (inverse) ? conj(second) : second;

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i10] *= shifts_0;
        arr[i11] *= shifts_1;
    }
};

template <class Precision, bool inverse = false> struct applyCSWAPFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;
    Kokkos::complex<Precision> shifts_2;

    applyCSWAPFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                      std::size_t num_qubits,
                      const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[2] - 1;
        rev_wire1 = num_qubits - wires[1] - 1;
        rev_wire2 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        if (rev_wire2 < rev_wire_min) {
            rev_wire_mid = rev_wire_min;
            rev_wire_min = rev_wire2;
        } else if (rev_wire2 > rev_wire_max) {
            rev_wire_mid = rev_wire_max;
            rev_wire_max = rev_wire2;
        } else {
            rev_wire_mid = rev_wire2;
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_mid);
        parity_hmiddle =
            fillLeadingOnes(rev_wire_mid + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i000 =
            ((k << 3U) & parity_high) | ((k << 2U) & parity_hmiddle) |
            ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const std::size_t i101 = i000 | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i110 = i000 | rev_wire2_shift | rev_wire1_shift;

        KE::swap(arr[i101], arr[i110]);
    }
};

template <class Precision, bool inverse = false> struct applyToffoliFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;
    Kokkos::complex<Precision> shifts_2;

    applyToffoliFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                        std::size_t num_qubits,
                        const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[2] - 1;
        rev_wire1 = num_qubits - wires[1] - 1;
        rev_wire2 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        if (rev_wire2 < rev_wire_min) {
            rev_wire_mid = rev_wire_min;
            rev_wire_min = rev_wire2;
        } else if (rev_wire2 > rev_wire_max) {
            rev_wire_mid = rev_wire_max;
            rev_wire_max = rev_wire2;
        } else {
            rev_wire_mid = rev_wire2;
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_mid);
        parity_hmiddle =
            fillLeadingOnes(rev_wire_mid + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i000 =
            ((k << 3U) & parity_high) | ((k << 2U) & parity_hmiddle) |
            ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const std::size_t i111 =
            i000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i110 = i000 | rev_wire2_shift | rev_wire1_shift;

        KE::swap(arr[i111], arr[i110]);
    }
};

template <class Precision, bool inverse = false> struct applyMultiRZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t wires_parity;

    Kokkos::complex<Precision> shift_0;
    Kokkos::complex<Precision> shift_1;

    applyMultiRZFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                        std::size_t num_qubits,
                        const std::vector<size_t> &wires,
                        const std::vector<Precision> &params) {

        const Precision &angle = params[0];
        const Kokkos::complex<Precision> first = Kokkos::complex<Precision>{
            std::cos(angle / 2), -std::sin(angle / 2)};
        const Kokkos::complex<Precision> second = Kokkos::complex<Precision>{
            std::cos(angle / 2), std::sin(angle / 2)};

        shift_0 = (inverse) ? conj(first) : first;
        shift_1 = (inverse) ? conj(second) : second;

        std::size_t wires_parity_ = 0U;
        for (size_t wire : wires) {
            wires_parity_ |=
                (static_cast<size_t>(1U) << (num_qubits - wire - 1));
        }

        wires_parity = wires_parity_;
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        arr[k] *= (Kokkos::Impl::bit_count(k & wires_parity) % 2 == 0)
                      ? shift_0
                      : shift_1;
    }
};

template <class Precision, bool adj = false>
struct applyGeneratorPhaseShiftFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    applyGeneratorPhaseShiftFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        arr[i0] = Kokkos::complex<Precision>{0.0, 0.0};
    }
};

template <class Precision, bool adj = false>
struct applyGeneratorIsingXXFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyGeneratorIsingXXFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        KE::swap(arr[i00], arr[i11]);
        KE::swap(arr[i10], arr[i01]);
    }
};

template <class Precision, bool adj = false>
struct applyGeneratorIsingXYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyGeneratorIsingXYFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        KE::swap(arr[i10], arr[i01]);
        arr[i00] = Kokkos::complex<Precision>{0.0, 0.0};
        arr[i11] = Kokkos::complex<Precision>{0.0, 0.0};
    }
};

template <class Precision, bool adj = false>
struct applyGeneratorIsingYYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyGeneratorIsingYYFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const auto v00 = arr[i00];
        arr[i00] = -arr[i11];
        arr[i11] = -v00;
        KE::swap(arr[i10], arr[i01]);
    }
};

template <class Precision, bool adj = false>
struct applyGeneratorIsingZZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyGeneratorIsingZZFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;

        arr[i10] *= -1;
        arr[i01] *= -1;
    }
};

template <class Precision, bool adj = false>
struct applyGeneratorSingleExcitationFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyGeneratorSingleExcitationFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] = ComplexPrecision{};
        arr[i01] *= ComplexPrecision{0, 1};
        arr[i10] *= ComplexPrecision{0, -1};
        arr[i11] = ComplexPrecision{};
        KE::swap(arr[i10], arr[i01]);
    }
};

template <class Precision, bool adj = false>
struct applyGeneratorSingleExcitationMinusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyGeneratorSingleExcitationMinusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;

        arr[i01] *= ComplexPrecision{0, 1};
        arr[i10] *= ComplexPrecision{0, -1};
        KE::swap(arr[i10], arr[i01]);
    }
};

template <class Precision, bool adj = false>
struct applyGeneratorSingleExcitationPlusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyGeneratorSingleExcitationPlusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] *= -1;
        arr[i01] *= ComplexPrecision{0, 1};
        arr[i10] *= ComplexPrecision{0, -1};
        arr[i11] *= -1;

        KE::swap(arr[i10], arr[i01]);
    }
};

template <class Precision, bool inverse = false>
struct applyGeneratorDoubleExcitationFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire3;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire3_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_min_mid;
    std::size_t rev_wire_max_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;
    Kokkos::complex<Precision> shifts_2;
    Kokkos::complex<Precision> shifts_3;

    applyGeneratorDoubleExcitationFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i0000 =
            ((k << 4U) & parity_high) | ((k << 3U) & parity_hmiddle) |
            ((k << 2U) & parity_middle) | ((k << 1U) & parity_lmiddle) |
            (k & parity_low);
        const std::size_t i0001 = i0000 | rev_wire0_shift;
        const std::size_t i0010 = i0000 | rev_wire1_shift;
        const std::size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i0100 = i0000 | rev_wire2_shift;
        const std::size_t i0101 = i0000 | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i0110 = i0000 | rev_wire2_shift | rev_wire1_shift;
        const std::size_t i0111 =
            i0000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1000 = i0000 | rev_wire3_shift;
        const std::size_t i1001 = i0000 | rev_wire3_shift | rev_wire0_shift;
        const std::size_t i1010 = i0000 | rev_wire3_shift | rev_wire1_shift;
        const std::size_t i1011 =
            i0000 | rev_wire3_shift | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;
        const std::size_t i1101 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i1110 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire1_shift;
        const std::size_t i1111 = i0000 | rev_wire3_shift | rev_wire2_shift |
                                  rev_wire1_shift | rev_wire0_shift;

        const Kokkos::complex<Precision> v3 = arr[i0011];
        const Kokkos::complex<Precision> v12 = arr[i1100];
        arr[i0000] = ComplexPrecision{};
        arr[i0001] = ComplexPrecision{};
        arr[i0010] = ComplexPrecision{};
        arr[i0011] = v12 * ComplexPrecision{0, -1};
        arr[i0100] = ComplexPrecision{};
        arr[i0101] = ComplexPrecision{};
        arr[i0110] = ComplexPrecision{};
        arr[i0111] = ComplexPrecision{};
        arr[i1000] = ComplexPrecision{};
        arr[i1001] = ComplexPrecision{};
        arr[i1010] = ComplexPrecision{};
        arr[i1011] = ComplexPrecision{};
        arr[i1100] = v3 * ComplexPrecision{0, 1};
        arr[i1101] = ComplexPrecision{};
        arr[i1110] = ComplexPrecision{};
        arr[i1111] = ComplexPrecision{};
    }
};

template <class Precision, bool inverse = false>
struct applyGeneratorDoubleExcitationMinusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire3;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire3_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_min_mid;
    std::size_t rev_wire_max_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;
    Kokkos::complex<Precision> shifts_2;
    Kokkos::complex<Precision> shifts_3;

    applyGeneratorDoubleExcitationMinusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i0000 =
            ((k << 4U) & parity_high) | ((k << 3U) & parity_hmiddle) |
            ((k << 2U) & parity_middle) | ((k << 1U) & parity_lmiddle) |
            (k & parity_low);
        const std::size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;

        arr[i0011] *= ComplexPrecision{0, 1};
        arr[i1100] *= ComplexPrecision{0, -1};
        KE::swap(arr[i1100], arr[i0011]);
    }
};

template <class Precision, bool inverse = false>
struct applyGeneratorDoubleExcitationPlusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire3;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire3_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_min_mid;
    std::size_t rev_wire_max_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;
    Kokkos::complex<Precision> shifts_2;
    Kokkos::complex<Precision> shifts_3;

    applyGeneratorDoubleExcitationPlusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i0000 =
            ((k << 4U) & parity_high) | ((k << 3U) & parity_hmiddle) |
            ((k << 2U) & parity_middle) | ((k << 1U) & parity_lmiddle) |
            (k & parity_low);
        const std::size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;

        arr[i0011] *= ComplexPrecision{0, -1};
        arr[i1100] *= ComplexPrecision{0, 1};
        KE::swap(arr[i1100], arr[i0011]);
    }
};

template <class Precision, bool adj = false>
struct applyGeneratorControlledPhaseShiftFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyGeneratorControlledPhaseShiftFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;

        arr[i00] = Kokkos::complex<Precision>{0.0, 0.0};
        arr[i01] = Kokkos::complex<Precision>{0.0, 0.0};
        arr[i10] = Kokkos::complex<Precision>{0.0, 0.0};
    }
};

template <class Precision, bool adj = false> struct applyGeneratorCRXFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyGeneratorCRXFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                             std::size_t num_qubits,
                             const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] = Kokkos::complex<Precision>{0.0, 0.0};
        arr[i01] = Kokkos::complex<Precision>{0.0, 0.0};
        KE::swap(arr[i10], arr[i11]);
    }
};

template <class Precision, bool adj = false> struct applyGeneratorCRYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyGeneratorCRYFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                             std::size_t num_qubits,
                             const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] = ComplexPrecision{};
        arr[i01] = ComplexPrecision{};

        const auto v0 = arr[i10];

        arr[i10] = ComplexPrecision{imag(arr[i11]), -real(arr[i11])};
        arr[i11] = ComplexPrecision{-imag(v0), real(v0)};
    }
};

template <class Precision, bool adj = false> struct applyGeneratorCRZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    applyGeneratorCRZFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                             std::size_t num_qubits,
                             const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] = ComplexPrecision{};
        arr[i01] = ComplexPrecision{};
        arr[i11] *= -1;
    }
};

template <class Precision, bool adj = false>
struct applyGeneratorMultiRZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t wires_parity;

    applyGeneratorMultiRZFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        wires_parity = static_cast<size_t>(0U);
        for (size_t wire : wires) {
            wires_parity |=
                (static_cast<size_t>(1U) << (num_qubits - wire - 1));
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        arr[k] *= static_cast<Precision>(
            1 - 2 * int(Kokkos::Impl::bit_count(k & wires_parity) % 2));
    }
};

template <class Precision, bool inverse = false> struct applyRotFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    Kokkos::complex<Precision> rot_mat_0b00;
    Kokkos::complex<Precision> rot_mat_0b10;
    Kokkos::complex<Precision> rot_mat_0b01;
    Kokkos::complex<Precision> rot_mat_0b11;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    applyRotFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                    std::size_t num_qubits, const std::vector<size_t> &wires,
                    const std::vector<Precision> &params) {
        const Precision phi = (inverse) ? -params[2] : params[0];
        const Precision theta = (inverse) ? -params[1] : params[1];
        const Precision omega = (inverse) ? -params[0] : params[2];
        const Precision c = std::cos(theta / 2);
        const Precision s = std::sin(theta / 2);
        const Precision p{phi + omega};
        const Precision m{phi - omega};

        auto imag = Kokkos::complex<Precision>(0, 1);
        rot_mat_0b00 = Kokkos::exp(static_cast<Precision>(p / 2) * (-imag)) * c;
        rot_mat_0b01 = -Kokkos::exp(static_cast<Precision>(m / 2) * imag) * s;
        rot_mat_0b10 = Kokkos::exp(static_cast<Precision>(m / 2) * (-imag)) * s;
        rot_mat_0b11 = Kokkos::exp(static_cast<Precision>(p / 2) * imag) * c;

        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const Kokkos::complex<Precision> v0 = arr[i0];
        const Kokkos::complex<Precision> v1 = arr[i1];
        arr[i0] = rot_mat_0b00 * v0 +
                  rot_mat_0b01 * v1; // NOLINT(readability-magic-numbers)
        arr[i1] = rot_mat_0b10 * v0 +
                  rot_mat_0b11 * v1; // NOLINT(readability-magic-numbers)
    }
};

} // namespace Functors
} // namespace Pennylane
