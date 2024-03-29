// Copyright 2021 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file UtilKokkos.hpp
 * Contains uncategorised utility functions.
 */
#pragma once

#include <cassert>
#include <climits>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#if __has_include(<version>)
#include <version>
#endif

#if __cpp_lib_math_constants >= 201907L
#include <numbers>
#endif

namespace Pennylane::Lightning_Kokkos::Util {
template <typename T> struct remove_complex { using type = T; };
template <typename T> struct remove_complex<std::complex<T>> {
    using type = T;
};
template <typename T> using remove_complex_t = typename remove_complex<T>::type;

template <typename T> struct is_complex : std::false_type {};

template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

template <typename T> constexpr bool is_complex_v = is_complex<T>::value;

/**
 * Utility hash function for complex vectors representing matrices.
 */
struct MatrixHasher {
    template <class Precision = double>
    std::size_t
    operator()(const std::vector<std::complex<Precision>> &matrix) const {
        std::size_t hash_val = matrix.size();
        for (const auto &c_val : matrix) {
            hash_val ^= std::hash<Precision>()(c_val.real()) ^
                        std::hash<Precision>()(c_val.imag());
        }
        return hash_val;
    }
};

struct BitSwapFunctor {
    BitSwapFunctor() = default;

    KOKKOS_INLINE_FUNCTION
    std::size_t operator()(const size_t &bits, const size_t &i,
                           const size_t &j) {
        size_t x = ((bits >> i) ^ (bits >> j)) & 1U;
        return bits ^ ((x << i) | (x << j));
    }
};

/**
 * @brief Swap bits in i-th and j-th position in place
 */
inline auto constexpr bitswap(size_t bits, const size_t i, const size_t j)
    -> size_t {
    size_t x = ((bits >> i) ^ (bits >> j)) & 1U;
    return bits ^ ((x << i) | (x << j));
}

/**
 * @brief Fill ones from LSB to rev_wire
 */
inline size_t fillTrailingOnes(size_t pos) {
    return (pos == 0) ? 0 : (~size_t(0) >> (CHAR_BIT * sizeof(size_t) - pos));
}

/**
 * @brief Fill ones from MSB to pos
 */
inline auto constexpr fillLeadingOnes(size_t pos) -> size_t {
    return (~size_t(0)) << pos;
}

/**
 * @brief Compile-time scalar real times complex number.
 *
 * @tparam U Precision of real value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Real scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstMult(U a, std::complex<T> b)
    -> std::complex<T> {
    return {a * b.real(), a * b.imag()};
}

/**
 * @brief Compile-time scalar complex times complex.
 *
 * @tparam U Precision of complex value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstMult(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}
template <class T, class U = T>
inline static constexpr auto ConstMultConj(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return {a.real() * b.real() + a.imag() * b.imag(),
            -a.imag() * b.real() + a.real() * b.imag()};
}

/**
 * @brief Compile-time scalar complex summation.
 *
 * @tparam T Precision of complex value `a` and result.
 * @tparam U Precision of complex value `b`.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstSum(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return a + b;
}

/**
 * @brief Return complex value 1+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{1,0}
 */
template <template <typename...> class ComplexT, typename T>
inline static constexpr auto ONE() -> ComplexT<T> {
    return {1, 0};
}

/**
 * @brief Return complex value 0+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,0}
 */
template <template <typename...> class ComplexT, typename T>
inline static constexpr auto ZERO() -> ComplexT<T> {
    return {0, 0};
}

/**
 * @brief Return complex value 0+1i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr ComplexT<T>{0,1}
 */
template <template <typename...> class ComplexT, typename T>
inline static constexpr auto IMAG() -> ComplexT<T> {
    return {0, 1};
}

template <template <typename...> class ComplexT, typename T>
inline static constexpr auto HALF() -> ComplexT<T> {
    return {0.5, 0};
}

template <template <typename...> class ComplexT, typename T>
inline static constexpr auto NEGONE() -> ComplexT<T> {
    return {-1, 0};
}

/**
 * @brief Returns sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T sqrt(2)
 */
template <template <typename...> class ComplexT, typename T>
inline static constexpr auto SQRT2() -> ComplexT<T> {
#if __cpp_lib_math_constants >= 201907L
    return std::numbers::sqrt2_v<T>;
#else
    if constexpr (std::is_same_v<T, float>) {
        return {0x1.6a09e6p+0F, 0}; // NOLINT: To be replaced in C++20
    } else {
        return {0x1.6a09e667f3bcdp+0, 0}; // NOLINT: To be replaced in C++20
    }
#endif
}

template <template <typename...> class ComplexT, typename T>
inline constexpr auto INVSQRT2() -> ComplexT<T> {
    return {1 / real(SQRT2<ComplexT, T>()), 0};
}

/**
 * @brief Calculates 2^n for some integer n > 0 using bitshifts.
 *
 * @param n the exponent
 * @return value of 2^n
 */
inline auto exp2(const size_t &n) -> size_t {
    return static_cast<size_t>(1) << n;
}

/**
 * @brief Log2 calculation.
 *
 * @param value Value to calculate for.
 * @return size_t
 */
inline auto log2(size_t value) -> size_t {
    return static_cast<size_t>(std::log2(value));
}

/**
 * @brief Calculates the decimal value for a qubit, assuming a big-endian
 * convention.
 *
 * @param qubitIndex the index of the qubit in the range [0, qubits)
 * @param qubits the number of qubits in the circuit
 * @return decimal value for the qubit at specified index
 */
inline auto maxDecimalForQubit(size_t qubitIndex, size_t qubits) -> size_t {
    assert(qubitIndex < qubits);
    return exp2(qubits - qubitIndex - 1);
}

/**
 * @brief Returns the number of wires supported by a given qubit gate.
 *
 * @tparam T Floating point precision type.
 * @param data Gate matrix data.
 * @return size_t Number of wires.
 */
template <class T> inline auto dimSize(const std::vector<T> &data) -> size_t {
    const size_t s = data.size();
    const auto s_sqrt = static_cast<size_t>(std::floor(std::sqrt(s)));

    if (s < 4) {
        throw std::invalid_argument("The dataset must be at least 2x2");
    }
    if (((s == 0) || (s & (s - 1)))) {
        throw std::invalid_argument("The dataset must be a power of 2");
    }
    if (s_sqrt * s_sqrt != s) {
        throw std::invalid_argument("The dataset must be a perfect square");
    }

    return static_cast<size_t>(log2(s_sqrt));
}

/**
 * @brief Streaming operator for vector data.
 *
 * @tparam T Vector data type.
 * @param os Output stream.
 * @param vec Vector data.
 * @return std::ostream&
 */
template <class T>
inline auto operator<<(std::ostream &os, const std::vector<T> &vec)
    -> std::ostream & {
    os << '[';
    if (!vec.empty()) {
        for (size_t i = 0; i < vec.size() - 1; i++) {
            os << vec[i] << ", ";
        }
        os << vec.back();
    }
    os << ']';
    return os;
}

/**
 * @brief Streaming operator for set data.
 *
 * @tparam T Vector data type.
 * @param os Output stream.
 * @param s Set data.
 * @return std::ostream&
 */
template <class T>
inline auto operator<<(std::ostream &os, const std::set<T> &s)
    -> std::ostream & {
    os << '{';
    for (const auto &e : s) {
        os << e << ",";
    }
    os << '}';
    return os;
}

/**
 * @brief Define linearly spaced data [start, end]
 *
 * @tparam T Data type.
 * @param start Start position.
 * @param end End position.
 * @param num_points Number of data-points in range.
 * @return std::vector<T>
 */
template <class T>
auto linspace(T start, T end, size_t num_points) -> std::vector<T> {
    std::vector<T> data(num_points);
    T step = (end - start) / (num_points - 1);
    for (size_t i = 0; i < num_points; i++) {
        data[i] = start + (step * i);
    }
    return data;
}

/**
 * @brief Determines the indices that would sort an array.
 *
 * @tparam T Vector data type.
 * @param arr Array to be inspected.
 * @param length Size of the array
 * @return a vector with indices that would sort the array.
 */
template <typename T>
inline auto sorting_indices(const T *arr, size_t length)
    -> std::vector<size_t> {
    std::vector<size_t> indices(length);
    iota(indices.begin(), indices.end(), 0);

    // indices will be sorted in accordance to the array provided.
    sort(indices.begin(), indices.end(),
         [&arr](size_t i1, size_t i2) { return arr[i1] < arr[i2]; });

    return indices;
}

/**
 * @brief Determines the indices that would sort a vector.
 *
 * @tparam T Array data type.
 * @param vec Vector to be inspected.
 * @return a vector with indices that would sort the vector.
 */
template <typename T>
inline auto sorting_indices(const std::vector<T> &vec) -> std::vector<size_t> {
    return sorting_indices(vec.data(), vec.size());
}

/**
 * @brief Determines the transposed index of a tensor stored linearly.
 *  This function assumes each axis will have a length of 2 (|0>, |1>).
 *
 * @param ind index after transposition.
 * @param new_axes new axes distribution.
 * @return unsigned int with the new transposed index.
 */
inline auto transposed_state_index(size_t ind,
                                   const std::vector<size_t> &new_axes)
    -> size_t {
    size_t new_index = 0;
    for (size_t axis : new_axes) {
        new_index += (ind % 2) << axis;
        ind /= 2;
    }
    return new_index;
}

/**
 * @brief Template for the transposition of state tensors,
 * axes are assumed to have a length of 2 (|0>, |1>).
 *
 * @tparam T Tensor data type.
 * @param tensor Tensor to be transposed.
 * @param new_axes new axes distribution.
 * @return Transposed Tensor.
 */
template <typename T>
auto transpose_state_tensor(const std::vector<T> &tensor,
                            const std::vector<size_t> &new_axes)
    -> std::vector<T> {
    std::vector<T> transposed_tensor(tensor.size());
    for (size_t ind = 0; ind < tensor.size(); ind++) {
        transposed_tensor[transposed_state_index(ind, new_axes)] = tensor[ind];
    }
    return transposed_tensor;
}

/**
 * @brief Exception for functions that are not yet implemented.
 *
 */
class NotImplementedException : public std::logic_error {
  public:
    /**
     * @brief Construct a NotImplementedException exception object.
     *
     * @param fname Function name to indicate not implemented.
     */
    explicit NotImplementedException(const std::string &fname)
        : std::logic_error(std::string("Function is not implemented. ") +
                           fname){};
};

/**
 * @brief Chunk the data using the requested chunk size.
 *
 * @tparam Container STL container type
 * @tparam T Data-type of STL container
 * @param data Data to chunk
 * @param chunk_size Chunk size to use.
 * @return Container<Container<T>> Container of {Containers of data with sizes
 * chunk_size}
 */
template <template <typename...> class Container, typename T>
auto chunkDataSize(const Container<T> &data, std::size_t chunk_size)
    -> Container<Container<T>> {
    Container<Container<T>> output;
    for (std::size_t chunk = 0; chunk < data.size(); chunk += chunk_size) {
        const auto chunk_end = std::min(data.size(), chunk + chunk_size);
        output.emplace_back(data.begin() + chunk, data.begin() + chunk_end);
    }
    return output;
}

/** Chunk the data into the requested number of chunks */

/**
 * @brief Chunk the data into the requested number of chunks.
 *
 * @tparam Container STL container type
 * @tparam T Data-type of STL container
 * @param data Data to chunk
 * @param num_chunks Chunk size to use.
 * @return Container<Container<T>> Container of num_chunks {Containers of data}
 */
template <template <typename...> class Container, typename T>
auto chunkData(const Container<T> &data, std::size_t num_chunks)
    -> Container<Container<T>> {
    const auto rem = data.size() % num_chunks;
    const auto div = static_cast<std::size_t>(data.size() / num_chunks);
    if (!div) { // Match chunks to available work
        return chunkDataSize(data, 1);
    }
    if (rem) { // We have an uneven split; ensure fair distribution
        auto output =
            chunkDataSize(Container<T>{data.begin(), data.end() - rem}, div);
        auto output_rem =
            chunkDataSize(Container<T>{data.end() - rem, data.end()}, div);
        for (std::size_t idx = 0; idx < output_rem.size(); idx++) {
            output[idx].insert(output[idx].end(), output_rem[idx].begin(),
                               output_rem[idx].end());
        }
        return output;
    }
    return chunkDataSize(data, div);
}

/**
 * @brief Define a hash function for std::pair
 */
struct PairHash {
    /**
     * @brief A hash function for std::pair
     *
     * @tparam T The type of the first element of the pair
     * @tparam U The type of the first element of the pair
     * @param p A pair to compute hash
     */
    template <typename T, typename U>
    size_t operator()(const std::pair<T, U> &p) const {
        return std::hash<T>()(p.first) ^ std::hash<U>()(p.second);
    }
};

/**
 * @brief Iterate over all enum values (if BEGIN and END are defined).
 *
 * @tparam T enum type
 * @tparam Func function to execute
 */
template <class T, class Func> void for_each_enum(Func &&func) {
    for (auto e = T::BEGIN; e != T::END;
         e = static_cast<T>(std::underlying_type_t<T>(e) + 1)) {
        func(e);
    }
}
template <class T, class U, class Func> void for_each_enum(Func &&func) {
    for (auto e1 = T::BEGIN; e1 != T::END;
         e1 = static_cast<T>(std::underlying_type_t<T>(e1) + 1)) {
        for (auto e2 = U::BEGIN; e2 != U::END;
             e2 = static_cast<U>(std::underlying_type_t<U>(e2) + 1)) {
            func(e1, e2);
        }
    }
}

/**
 * @brief Generate indices for applying operations.
 *
 * This method will return the statevector indices participating in the
 * application of a gate to a given set of qubits.
 *
 * @param qubitIndices Indices of the qubits to apply operations.
 * @param num_qubits Number of qubits in register.
 * @return std::vector<size_t>
 */

inline auto
getIndicesAfterExclusion(const std::vector<size_t> &indicesToExclude,
                         size_t num_qubits) -> std::vector<size_t> {
    std::vector<size_t> indices;
    for (size_t i = 0; i < num_qubits; i++) {
        indices.emplace_back(i);
    }

    for (size_t j = 0; j < indicesToExclude.size(); j++) {

        const size_t excludedIndex = indicesToExclude[j];

        for (size_t i = 0; i < indices.size(); i++) {
            if (excludedIndex == indices[i])
                indices.erase(indices.begin() + i);
        }
    }
    return indices;
}

/**
 * @brief Generate indices for applying operations.
 *
 * This method will return the statevector indices participating in the
 * application of a gate to a given set of qubits.
 *
 * @param qubitIndices Indices of the qubits to apply operations.
 * @param num_qubits Number of qubits in register.
 * @return std::vector<size_t>
 */

inline auto generateBitsPatterns(const std::vector<size_t> &qubitIndices,
                                 size_t num_qubits) -> std::vector<size_t> {

    std::vector<size_t> indices;
    indices.reserve(exp2(qubitIndices.size()));
    indices.emplace_back(0);

    for (size_t index_it0 = 0; index_it0 < qubitIndices.size(); index_it0++) {
        size_t index_it = qubitIndices.size() - 1 - index_it0;
        const size_t value =
            maxDecimalForQubit(qubitIndices[index_it], num_qubits);

        const size_t currentSize = indices.size();
        for (size_t j = 0; j < currentSize; j++) {
            indices.emplace_back(indices[j] + value);
        }
    }
    return indices;
}

/**
 * @brief Streaming operator for Kokkos::InitializationSettings objects.
 *
 * @param os Output stream.
 * @param args Kokkos::InitializationSettings object.
 * @return std::ostream&
 */
inline auto operator<<(std::ostream &os,
                       const Kokkos::InitializationSettings &args)
    -> std::ostream & {
    os << "InitializationSettings:\n";
    os << "num_threads = " << args.get_num_threads() << '\n';
    os << "device_id = " << args.get_device_id() << '\n';
    os << "map_device_id_by = " << args.get_map_device_id_by() << '\n';
    os << "disable_warnings = " << args.get_disable_warnings() << '\n';
    os << "print_configuration = " << args.get_print_configuration() << '\n';
    os << "tune_internals = " << args.get_tune_internals() << '\n';
    os << "tools_libs = " << args.get_tools_libs() << '\n';
    os << "tools_help = " << args.get_tools_help() << '\n';
    os << "tools_args = " << args.get_tools_args();
    return os;
}

} // namespace Pennylane::Lightning_Kokkos::Util
