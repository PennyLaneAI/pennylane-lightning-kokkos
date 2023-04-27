// Copyright 2018-2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cmath>
#include <complex>
#include <vector>

#include "UtilKokkos.hpp"
#include <Kokkos_Core.hpp>

/// @cond DEV
namespace {
namespace Util = Pennylane::Lightning_Kokkos::Util;
using namespace Util;
} // namespace
/// @endcond

namespace Pennylane::Gates {

/**
 * @brief Create a matrix representation of the PauliX gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of PauliX data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getIdentity() -> std::vector<ComplexT<T>> {
    return {Util::ONE<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the PauliX gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of PauliX data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getPauliX() -> std::vector<ComplexT<T>> {
    return {Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ONE<ComplexT, T>(), Util::ZERO<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the PauliY gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of PauliY data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getPauliY() -> std::vector<ComplexT<T>> {
    return {Util::ZERO<ComplexT, T>(), -Util::IMAG<ComplexT, T>(),
            Util::IMAG<ComplexT, T>(), Util::ZERO<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the PauliZ gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of PauliZ data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getPauliZ() -> std::vector<ComplexT<T>> {
    return {Util::ONE<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), -Util::ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the Hadamard gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of Hadamard data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getHadamard() -> std::vector<ComplexT<T>> {
    return {Util::INVSQRT2<ComplexT, T>(), Util::INVSQRT2<ComplexT, T>(),
            Util::INVSQRT2<ComplexT, T>(), -Util::INVSQRT2<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the S gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of S gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getS() -> std::vector<ComplexT<T>> {
    return {Util::ONE<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::IMAG<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the T gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of T gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getT() -> std::vector<ComplexT<T>> {
    return {
        Util::ONE<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(),
        Util::ConstMult(Util::SQRT2<decltype(Util::ONE<ComplexT, T>().x)>() / 2,
                        Util::ConstSum(Util::ONE<ComplexT, T>(),
                                       -Util::IMAG<ComplexT, T>()))};
}

/**
 * @brief Create a matrix representation of the CNOT gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of CNOT gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getCNOT() -> std::vector<ComplexT<T>> {
    return {Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the SWAP gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of SWAP gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getSWAP() -> std::vector<ComplexT<T>> {
    return {Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the CZ gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of SWAP gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getCY() -> std::vector<ComplexT<T>> {
    return {Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), -Util::IMAG<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::IMAG<ComplexT, T>(), Util::ZERO<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the CZ gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of SWAP gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getCZ() -> std::vector<ComplexT<T>> {
    return {Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), -Util::ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the CSWAP gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of CSWAP gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getCSWAP() -> std::vector<ComplexT<T>> {
    return {Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the Toffoli gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of Toffoli gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getToffoli() -> std::vector<ComplexT<T>> {
    return {Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the Phase-shift gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return Phase-shift gate
 * data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getPhaseShift(T angle) -> std::vector<ComplexT<T>> {
    return {Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            {std::cos(angle), std::sin(angle)}};
}

/**
 * @brief Create a matrix representation of the Phase-shift gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return Phase-shift gate
 * data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getPhaseShift(const std::vector<T> &params)
    -> std::vector<ComplexT<T>> {
    return getPhaseShift<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the RX gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RX gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getRX(T angle) -> std::vector<ComplexT<T>> {
    const ComplexT<T> c{std::cos(angle / 2), 0};
    const ComplexT<T> js{0, -std::sin(angle / 2)};
    return {c, js, js, c};
}

/**
 * @brief Create a matrix representation of the RX gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return RX gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getRX(const std::vector<T> &params) -> std::vector<ComplexT<T>> {
    return getRX<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the RY gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RY gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getRY(T angle) -> std::vector<ComplexT<T>> {
    const ComplexT<T> c{std::cos(angle / 2), 0};
    const ComplexT<T> s{std::sin(angle / 2), 0};
    return {c, -s, s, c};
}

/**
 * @brief Create a matrix representation of the RY gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return RY gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getRY(const std::vector<T> &params) -> std::vector<ComplexT<T>> {
    return getRY<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the RZ gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RZ gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getRZ(T angle) -> std::vector<ComplexT<T>> {
    return {{std::cos(-angle / 2), std::sin(-angle / 2)},
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            {std::cos(angle / 2), std::sin(angle / 2)}};
}

/**
 * @brief Create a matrix representation of the RZ gate data in row-major
 * format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return RZ gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getRZ(const std::vector<T> &params) -> std::vector<ComplexT<T>> {
    return getRZ<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the Rot gate data in row-major
format.
 *
 * The gate is defined as:
 * \f$\begin{split}Rot(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)=
\begin{bmatrix}
e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
\end{bmatrix}.\end{split}\f$
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param phi \f$\phi\f$ shift angle.
 * @param theta \f$\theta\f$ shift angle.
 * @param omega \f$\omega\f$ shift angle.
 * @return std::vector<ComplexT<T>> Return Rot gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getRot(T phi, T theta, T omega) -> std::vector<ComplexT<T>> {
    const T c = std::cos(theta / 2);
    const T s = std::sin(theta / 2);
    const T p{phi + omega};
    const T m{phi - omega};

    return {ComplexT<T>{std::cos(p / 2) * c, -std::sin(p / 2) * c},
            ComplexT<T>{-std::cos(m / 2) * s, -std::sin(m / 2) * s},
            ComplexT<T>{std::cos(m / 2) * s, -std::sin(m / 2) * s},
            ComplexT<T>{std::cos(p / 2) * c, std::sin(p / 2) * c}};
}

/**
 * @brief Create a matrix representation of the Rot gate data in row-major
format.
 *
 * The gate is defined as:
 * \f$\begin{split}Rot(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)=
\begin{bmatrix}
e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
\end{bmatrix}.\end{split}\f$
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of gate data. Values are expected in order of
\f$[\phi, \theta, \omega]\f$.
 * @return std::vector<ComplexT<T>> Return Rot gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getRot(const std::vector<T> &params) -> std::vector<ComplexT<T>> {
    return getRot<ComplexT<T>>(params[0], params[1], params[2]);
}

/**
 * @brief Create a matrix representation of the controlled RX gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RX gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRX(T angle) -> std::vector<ComplexT<T>> {
    const auto rx{getRX<ComplexT<T>>(angle)};
    return {Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            rx[0],
            rx[1],
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            rx[2],
            rx[3]};
}

/**
 * @brief Create a matrix representation of the controlled RX gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return RX gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRX(const std::vector<T> &params) -> std::vector<ComplexT<T>> {
    return getCRX<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled RY gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RY gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRY(T angle) -> std::vector<ComplexT<T>> {
    const auto ry{getRY<ComplexT<T>>(angle)};
    return {Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            ry[0],
            ry[1],
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            ry[2],
            ry[3]};
}

/**
 * @brief Create a matrix representation of the controlled RY gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return RY gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRY(const std::vector<T> &params) -> std::vector<ComplexT<T>> {
    return getCRY<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled RZ gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RZ gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRZ(T angle) -> std::vector<ComplexT<T>> {
    const ComplexT<T> first{std::cos(-angle / 2), std::sin(-angle / 2)};
    const ComplexT<T> second{std::cos(angle / 2), std::sin(angle / 2)};
    return {Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            first,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            second};
}

/**
 * @brief Create a matrix representation of the controlled RZ gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return RZ gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRZ(const std::vector<T> &params) -> std::vector<ComplexT<T>> {
    return getCRZ<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled Rot gate data in
row-major format.
 *
 * @see `getRot<T,U>(U phi, U theta, U omega)`.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRot(T phi, T theta, T omega) -> std::vector<ComplexT<T>> {
    const auto rot{std::move(getRot<ComplexT<T>>(phi, theta, omega))};
    return {Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            rot[0],
            rot[1],
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            rot[2],
            rot[3]};
}

/**
 * @brief Create a matrix representation of the controlled Rot gate data in
row-major format.
 *
 * @see `getRot<T,U>(const std::vector<T> &params)`.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRot(const std::vector<T> &params) -> std::vector<T> {
    return getCRot<T>(params[0], params[1], params[2]);
}

/**
 * @brief Create a matrix representation of the controlled phase-shift gate
data in row-major format.
 *
 * @see `getPhaseShift<T,U>(U angle)`.
 */
template <template <typename...> class ComplexT, typename T>
static auto getControlledPhaseShift(T angle) -> std::vector<ComplexT<T>> {
    return {Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(), {std::cos(angle), std::sin(angle)}};
}

/**
 * @brief Create a matrix representation of the controlled phase-shift gate
data in row-major format.
 *
 * @see `getPhaseShift<T,U>(const std::vector<T> &params)`.
 */
template <template <typename...> class ComplexT, typename T>
static auto getControlledPhaseShift(const std::vector<T> &params)
    -> std::vector<ComplexT<T>> {
    return getControlledPhaseShift<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return single excitation rotation
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getSingleExcitation(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> s{std::sin(p2), 0};
    return {Util::ONE<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            c,
            s,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            -s,
            c,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return single excitation rotation
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getSingleExcitation(const std::vector<T> &params)
    -> std::vector<ComplexT<T>> {
    return getSingleExcitation<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the SingleExcitation
 * generator data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorSingleExcitation()
    -> std::vector<ComplexT<T>> {
    return {
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::IMAG<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), -Util::IMAG<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getSingleExcitationMinus(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> e = Kokkos::Experimental::exp(ComplexT<T>(0, -p2));
    const ComplexT<T> c{Kokkos::Experimental::cos(p2), 0};
    const ComplexT<T> s{Kokkos::Experimental::sin(p2), 0};
    return {e,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            c,
            s,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            -s,
            c,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            e};
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getSingleExcitationMinus(const std::vector<T> &params)
    -> std::vector<ComplexT<T>> {
    return getSingleExcitationMinus<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the SingleExcitation Minus
 * generator data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorSingleExcitationMinus()
    -> std::vector<ComplexT<T>> {
    return {
        Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::IMAG<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), -Util::IMAG<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getSingleExcitationPlus(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> e = Kokkos::Experimental::exp(ComplexT<T>(0, p2));
    const ComplexT<T> c{Kokkos::Experimental::cos(p2), 0};
    const ComplexT<T> s{Kokkos::Experimental::sin(p2), 0};
    return {e,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            c,
            s,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            -s,
            c,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            e};
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getSingleExcitationPlus(const std::vector<T> &params)
    -> std::vector<ComplexT<T>> {
    return getSingleExcitationPlus<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the SingleExcitation Plus
 * generator data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorSingleExcitationPlus()
    -> std::vector<ComplexT<T>> {
    return {
        -Util::ONE<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::IMAG<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), -Util::IMAG<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), -Util::ONE<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return double excitation rotation
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getDoubleExcitation(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> s{std::sin(p2), 0};
    std::vector<ComplexT<T>> mat(256, Util::ZERO<ComplexT, T>());
    mat[0] = Util::ONE<ComplexT, T>();
    mat[17] = Util::ONE<ComplexT, T>();
    mat[34] = Util::ONE<ComplexT, T>();
    mat[51] = c;
    mat[60] = s;
    mat[68] = Util::ONE<ComplexT, T>();
    mat[85] = Util::ONE<ComplexT, T>();
    mat[102] = Util::ONE<ComplexT, T>();
    mat[119] = Util::ONE<ComplexT, T>();
    mat[136] = Util::ONE<ComplexT, T>();
    mat[153] = Util::ONE<ComplexT, T>();
    mat[170] = Util::ONE<ComplexT, T>();
    mat[187] = Util::ONE<ComplexT, T>();
    mat[195] = -s;
    mat[204] = c;
    mat[221] = Util::ONE<ComplexT, T>();
    mat[238] = Util::ONE<ComplexT, T>();
    mat[255] = Util::ONE<ComplexT, T>();
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return double excitation rotation
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getDoubleExcitation(const std::vector<T> &params)
    -> std::vector<ComplexT<T>> {
    return getDoubleExcitation<ComplexT, T>(params.front());
}

/**
 * @brief Create a matrix representation of the DoubleExcitation
 * generator data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorDoubleExcitation()
    -> std::vector<ComplexT<T>> {
    std::vector<ComplexT<T>> mat(256, Util::ZERO<ComplexT, T>());
    mat[60] = Util::IMAG<ComplexT, T>();
    mat[195] = -Util::IMAG<ComplexT, T>();
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getDoubleExcitationMinus(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> e = Kokkos::Experimental::exp(ComplexT<T>(0, -p2));
    const ComplexT<T> c{Kokkos::Experimental::cos(p2), 0};
    const ComplexT<T> s{Kokkos::Experimental::sin(p2), 0};
    std::vector<ComplexT<T>> mat(256, Util::ZERO<ComplexT, T>());
    mat[0] = e;
    mat[17] = e;
    mat[34] = e;
    mat[51] = c;
    mat[60] = s;
    mat[68] = e;
    mat[85] = e;
    mat[102] = e;
    mat[119] = e;
    mat[136] = e;
    mat[153] = e;
    mat[170] = e;
    mat[187] = e;
    mat[195] = -s;
    mat[204] = c;
    mat[221] = e;
    mat[238] = e;
    mat[255] = e;
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getDoubleExcitationMinus(const std::vector<T> &params)
    -> std::vector<ComplexT<T>> {
    return getDoubleExcitationMinus<ComplexT<T>>(params.front());
}

/**
 * @brief Create a matrix representation of the DoubleExcitation Minus
 * generator data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorDoubleExcitationMinus()
    -> std::vector<ComplexT<T>> {
    std::vector<ComplexT<T>> mat(256, Util::ZERO<ComplexT, T>());
    mat[0] = Util::ONE<ComplexT, T>();
    mat[17] = Util::ONE<ComplexT, T>();
    mat[34] = Util::ONE<ComplexT, T>();
    mat[60] = Util::IMAG<ComplexT, T>();
    mat[68] = Util::ONE<ComplexT, T>();
    mat[85] = Util::ONE<ComplexT, T>();
    mat[102] = Util::ONE<ComplexT, T>();
    mat[119] = Util::ONE<ComplexT, T>();
    mat[136] = Util::ONE<ComplexT, T>();
    mat[153] = Util::ONE<ComplexT, T>();
    mat[170] = Util::ONE<ComplexT, T>();
    mat[187] = Util::ONE<ComplexT, T>();
    mat[195] = -Util::IMAG<ComplexT, T>();
    mat[221] = Util::ONE<ComplexT, T>();
    mat[238] = Util::ONE<ComplexT, T>();
    mat[255] = Util::ONE<ComplexT, T>();
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getDoubleExcitationPlus(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> e = Kokkos::Experimental::exp(ComplexT<T>(0, p2));
    const ComplexT<T> c{Kokkos::Experimental::cos(p2), 0};
    const ComplexT<T> s{Kokkos::Experimental::sin(p2), 0};
    std::vector<ComplexT<T>> mat(256, Util::ZERO<ComplexT, T>());
    mat[0] = e;
    mat[17] = e;
    mat[34] = e;
    mat[51] = c;
    mat[60] = s;
    mat[68] = e;
    mat[85] = e;
    mat[102] = e;
    mat[119] = e;
    mat[136] = e;
    mat[153] = e;
    mat[170] = e;
    mat[187] = e;
    mat[195] = -s;
    mat[204] = c;
    mat[221] = e;
    mat[238] = e;
    mat[255] = e;
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getDoubleExcitationPlus(const std::vector<T> &params)
    -> std::vector<ComplexT<T>> {
    return getDoubleExcitationPlus<ComplexT, T>(params.front());
}

/**
 * @brief Create a matrix representation of the DoubleExcitation Plus
 * generator data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorDoubleExcitationPlus()
    -> std::vector<ComplexT<T>> {
    std::vector<ComplexT<T>> mat(256, Util::ZERO<ComplexT, T>());
    mat[0] = -Util::ONE<ComplexT, T>();
    mat[17] = -Util::ONE<ComplexT, T>();
    mat[34] = -Util::ONE<ComplexT, T>();
    mat[60] = Util::IMAG<ComplexT, T>();
    mat[68] = -Util::ONE<ComplexT, T>();
    mat[85] = -Util::ONE<ComplexT, T>();
    mat[102] = -Util::ONE<ComplexT, T>();
    mat[119] = -Util::ONE<ComplexT, T>();
    mat[136] = -Util::ONE<ComplexT, T>();
    mat[153] = -Util::ONE<ComplexT, T>();
    mat[170] = -Util::ONE<ComplexT, T>();
    mat[187] = -Util::ONE<ComplexT, T>();
    mat[195] = -Util::IMAG<ComplexT, T>();
    mat[221] = -Util::ONE<ComplexT, T>();
    mat[238] = -Util::ONE<ComplexT, T>();
    mat[255] = -Util::ONE<ComplexT, T>();
    return mat;
}

/**
 * @brief Create a matrix representation of the Ising XX coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return Ising XX coupling
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getIsingXX(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> neg_is{0, -std::sin(p2)};
    return {c,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            neg_is,
            Util::ZERO<ComplexT, T>(),
            c,
            neg_is,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            neg_is,
            c,
            Util::ZERO<ComplexT, T>(),
            neg_is,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            c};
}

/**
 * @brief Create a matrix representation of the Ising XX coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return Ising XX coupling
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getIsingXX(const std::vector<T> &params)
    -> std::vector<ComplexT<T>> {
    return getIsingXX<ComplexT, T>(params.front());
}

/**
 * @brief Create a matrix representation of the Ising XX generator
 * data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<CFP_t>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorIsingXX() -> std::vector<ComplexT<T>> {
    return {
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising YY coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return Ising YY coupling
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getIsingYY(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> pos_is{0, std::sin(p2)};
    const ComplexT<T> neg_is{0, -std::sin(p2)};
    return {c,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            pos_is,
            Util::ZERO<ComplexT, T>(),
            c,
            neg_is,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            neg_is,
            c,
            Util::ZERO<ComplexT, T>(),
            pos_is,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            c};
}

/**
 * @brief Create a matrix representation of the Ising YY coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return Ising YY coupling
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getIsingYY(const std::vector<T> &params)
    -> std::vector<ComplexT<T>> {
    return getIsingYY<ComplexT, T>(params.front());
}

/**
 * @brief Create a matrix representation of the Ising YY generator
 * data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<CFP_t>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorIsingYY() -> std::vector<ComplexT<T>> {
    return {
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), -Util::ONE<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        -Util::ONE<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising ZZ coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return Ising ZZ coupling
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getIsingZZ(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> neg_e = Kokkos::Experimental::exp(ComplexT<T>(0, -p2));
    const ComplexT<T> pos_e = Kokkos::Experimental::exp(ComplexT<T>(0, p2));
    return {neg_e,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            pos_e,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            pos_e,
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            Util::ZERO<ComplexT, T>(),
            neg_e};
}

/**
 * @brief Create a matrix representation of the Ising ZZ coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT<T>> Return Ising ZZ coupling
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getIsingZZ(const std::vector<T> &params)
    -> std::vector<ComplexT<T>> {
    return getIsingZZ<ComplexT, T>(params.front());
}

/**
 * @brief Create a matrix representation of the Ising ZZ generator
 * data in row-major format.
 *
 * @tparam ComplexT<T> Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorIsingZZ() -> std::vector<ComplexT<T>> {
    return {
        -Util::ONE<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ONE<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ONE<ComplexT, T>(),  Util::ZERO<ComplexT, T>(),

        Util::ZERO<ComplexT, T>(), Util::ZERO<ComplexT, T>(),
        Util::ZERO<ComplexT, T>(), -Util::ONE<ComplexT, T>(),
    };
}

} // namespace Pennylane::Gates
