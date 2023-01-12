// Copyright 2022 Xanadu Quantum Technologies Inc.

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
 * @file Bindings.hpp
 * Defines operations to export to Python and other utility functions
 * interfacing with Pybind11
 */
#pragma once

#include "StateVectorKokkos.hpp"
#include "pybind11/pybind11.h"

#include <iostream>
#include <string>

namespace {
namespace py = pybind11;

/**
 * @brief Provide Kokkos backend.
 */
auto getConfig() -> pybind11::dict {
    using namespace pybind11::literals;

    // Redirect std::cout to string variable
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Kokkos::print_configuration(std::cout, true);

    std::cout.rdbuf(old);
    std::string config_str = buffer.str();

    return pybind11::dict("All_Info"_a = config_str);
}

} // namespace
