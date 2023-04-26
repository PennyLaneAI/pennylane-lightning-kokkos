#pragma once

#include <Kokkos_Core.hpp>

#include "UtilKokkos.hpp"

namespace {
using namespace Pennylane::Lightning_Kokkos::Util;
}

namespace Pennylane::Functors {

template <class Precision> struct getExpectationValueIdentityFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    getExpectationValueIdentityFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> arr_,
        [[maybe_unused]] std::size_t num_qubits,
        [[maybe_unused]] const std::vector<size_t> &wires) {
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, Precision &expval) const {
        expval += real(conj(arr[k]) * arr[k]);
    }
};

template <class Precision> struct getExpectationValuePauliXFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValuePauliXFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> arr_, std::size_t num_qubits,
        const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, Precision &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;

        expval += real(conj(arr[i0]) * arr[i1]);
        expval += real(conj(arr[i1]) * arr[i0]);
    }
};

template <class Precision> struct getExpectationValuePauliYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValuePauliYFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> arr_, std::size_t num_qubits,
        const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, Precision &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr[i0];
        const auto v1 = arr[i1];

        expval += real(conj(arr[i0]) *
                       Kokkos::complex<Precision>{imag(v1), -real(v1)});
        expval += real(conj(arr[i1]) *
                       Kokkos::complex<Precision>{-imag(v0), real(v0)});
    }
};

template <class Precision> struct getExpectationValuePauliZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValuePauliZFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> arr_, std::size_t num_qubits,
        const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, Precision &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        expval += real(conj(arr[i1]) * (-arr[i1]));
        expval += real(conj(arr[i0]) * (arr[i0]));
    }
};

template <class Precision> struct getExpectationValueHadamardFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValueHadamardFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> arr_, std::size_t num_qubits,
        const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, Precision &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const Kokkos::complex<Precision> v0 = arr[i0];
        const Kokkos::complex<Precision> v1 = arr[i1];

        expval += real(M_SQRT1_2 *
                       (conj(arr[i0]) * (v0 + v1) + conj(arr[i1]) * (v0 - v1)));
    }
};

template <class Precision> struct getExpectationValueSingleQubitOpFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;
    Kokkos::View<Kokkos::complex<Precision> *> matrix;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValueSingleQubitOpFunctor(
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
    void operator()(const std::size_t k, Precision &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;

        expval += real(
            conj(arr[i0]) * (matrix[0B00] * arr[i0] + matrix[0B01] * arr[i1]) +
            conj(arr[i1]) * (matrix[0B10] * arr[i0] + matrix[0B11] * arr[i1]));
    }
};

template <class Precision> struct getExpectationValueTwoQubitOpFunctor {

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

    getExpectationValueTwoQubitOpFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits,
        const Kokkos::View<Kokkos::complex<Precision> *> &matrix_,
        const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1;

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
    void operator()(const std::size_t k, Precision &expval) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        expval +=
            real(conj(arr[i00]) *
                     (matrix[0B0000] * arr[i00] + matrix[0B0001] * arr[i01] +
                      matrix[0B0010] * arr[i10] + matrix[0B0011] * arr[i11]) +
                 conj(arr[i01]) *
                     (matrix[0B0100] * arr[i00] + matrix[0B0101] * arr[i01] +
                      matrix[0B0110] * arr[i10] + matrix[0B0111] * arr[i11]) +
                 conj(arr[i10]) *
                     (matrix[0B1000] * arr[i00] + matrix[0B1001] * arr[i01] +
                      matrix[0B1010] * arr[i10] + matrix[0B1011] * arr[i11]) +
                 conj(arr[i11]) *
                     (matrix[0B1100] * arr[i00] + matrix[0B1101] * arr[i01] +
                      matrix[0B1110] * arr[i10] + matrix[0B1111] * arr[i11]));
    }
};

template <class Precision, bool inverse = false>
struct getExpectationValueMultiQubitOpFunctor {

    using KokkosComplexVector = Kokkos::View<Kokkos::complex<Precision> *>;
    using KokkosSizeTVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosSizeTVector indices;
    KokkosSizeTVector wires;
    KokkosComplexVector coeffs_in;
    std::size_t dim;
    std::size_t num_qubits;
    static const BitSwapFunctor bsf;

    getExpectationValueMultiQubitOpFunctor(KokkosComplexVector arr_,
                                           std::size_t num_qubits_,
                                           const KokkosComplexVector matrix_,
                                           const KokkosSizeTVector wires_) {
        dim = 1U << wires_.size();
        indices = KokkosSizeTVector("indices", dim);
        coeffs_in = KokkosComplexVector("coeffs_in", dim);
        num_qubits = num_qubits_;
        wires = wires_;
        matrix = matrix_;
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t kp, Precision &expval) const {
        const std::size_t k = kp * dim;
        using Pennylane::Lightning_Kokkos::Util::bitswap;
        for (std::size_t inner_idx = 0; inner_idx < dim; inner_idx++) {
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
            const std::size_t base_idx = i * dim;

            Kokkos::complex<Precision> arr_idx_new = 0.0;
            for (size_t j = 0; j < dim; j++) {
                arr_idx_new += (matrix[base_idx + j] * coeffs_in[j]);
            }

            expval += real(arr_idx_new * conj(arr[idx]));
        }
    }
};

template <class Precision> struct getExpectationValueSparseFunctor {

    using KokkosComplexVector = Kokkos::View<Kokkos::complex<Precision> *>;
    using KokkosSizeTVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector data;
    KokkosSizeTVector indices;
    KokkosSizeTVector indptr;
    std::size_t length;

    getExpectationValueSparseFunctor(KokkosComplexVector arr_,
                                     const KokkosComplexVector data_,
                                     const KokkosSizeTVector indices_,
                                     const KokkosSizeTVector indptr_) {
        length = indices_.size();
        indices = indices_;
        indptr = indptr_;
        data = data_;
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t row, Precision &expval) const {
        for (size_t j = indptr[row]; j < indptr[row + 1]; j++) {
            expval += real(conj(arr[row]) * data[j] * arr[indices[j]]);
        }
    }
};

} // namespace Pennylane::Functors
