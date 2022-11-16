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
 * @file KokkosUtil.hpp
 * Kokkos utility functions.
 */

#include <Kokkos_Core.hpp>
#include <bits/stdc++.h>

using namespace std;
/**
 * @brief Streaming operator for Kokkos::InitArguments structs.
 *
 * @param os Output stream.
 * @param args Kokkos::InitArguments struct.
 * @return std::ostream&
 */
std::ostream &operator<<(std::ostream &os, const Kokkos::InitArguments &args) {
    os << "<InitArguments with\n";
    os << "num_threads = " << args.num_threads << '\n';
    os << "num_numa = " << args.num_numa << '\n';
    os << "device_id = " << args.device_id << '\n';
    os << "ndevices = " << args.ndevices << '\n';
    os << "skip_device = " << args.skip_device << '\n';
    os << "disable_warnings = " << args.disable_warnings << ">";
    return os;
}

/**
 * @brief Returns a printable representation of Kokkos::InitArguments structs.
 *
 * @param args Kokkos::InitArguments struct.
 * @return std::string
 */
std::string repr_InitArguments(const Kokkos::InitArguments &args) {
    std::ostringstream args_stream;
    args_stream << args;
    return args_stream.str();
}

/**
 * @brief Prints args representation of Kokkos::InitArguments structs.
 *
 * @param args Kokkos::InitArguments struct.
 */
void print_InitArguments(const Kokkos::InitArguments &args) {
    std::cout << args << '\n';
}