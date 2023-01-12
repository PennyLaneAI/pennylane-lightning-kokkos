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

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

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

    // Split string into string vector
    std::vector<std::string> config_str_vector;
    std::istringstream input0(config_str);

    for (std::string s0; std::getline(input0, s0, '\n');) {
        std::istringstream input1(s0);
        for (std::string s1; std::getline(input1, s1, ' ');) {
            std::istringstream input2(s1);
            for (std::string s2; std::getline(input2, s2, ':');) {
                config_str_vector.push_back(s2);
            }
        }
    }

    // Get kokkos info.
    std::string backend, kokkos_version, device, compiler;

    for (std::vector<std::string>::iterator it = config_str_vector.begin();
         it != config_str_vector.end(); ++it) {
        if (*it == "Runtime")
            backend = *(std::prev(it, 1));

        if (*it == "Version")
            kokkos_version = *(std::next(it, 1));

        if (*it == "Device")
            device = *(std::next(it, 1));

        if (*it == "Compiler")
            compiler = *(std::next(it, 1)) + " " + *(std::next(it, 2));
    }

    return pybind11::dict("All_Info"_a = config_str, "Backend"_a = backend,
                          "Compiler"_a = compiler,
                          "Kokkos_Version"_a = kokkos_version,
                          "Device_Arch"_a = device);
}

} // namespace
