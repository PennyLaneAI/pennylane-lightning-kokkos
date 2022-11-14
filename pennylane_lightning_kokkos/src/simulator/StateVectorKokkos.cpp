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

#include "StateVectorKokkos.hpp"

// explicit instantiation
template class Pennylane::StateVectorKokkos<float>;
template class Pennylane::StateVectorKokkos<double>;

std::string repr_InitArguments(const Kokkos::InitArguments &a) {
    std::string str;
    str = "<example.InitArguments with";
    str += "\n num_threads = " + std::to_string(a.num_threads);
    str += "\n num_numa = " + std::to_string(a.num_numa);
    str += "\n device_id = " + std::to_string(a.device_id);
    str += "\n ndevices = " + std::to_string(a.ndevices);
    str += "\n skip_device = " + std::to_string(a.skip_device);
    str += "\n disable_warnings = " + std::to_string(a.disable_warnings) += ">";
    return str;
}

void print_InitArguments(const Kokkos::InitArguments &a) {
    std::cout << repr_InitArguments(a) << '\n';
}