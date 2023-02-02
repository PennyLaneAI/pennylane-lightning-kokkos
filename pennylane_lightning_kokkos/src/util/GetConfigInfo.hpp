// Copyright 2023 Xanadu Quantum Technologies Inc.

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
 * @file GetConfigInfo.hpp
 * Defines operations to store information from Kokkos::print_configuration in
 * a unordered_map variable.
 */
#pragma once
#include "Kokkos_Core.hpp"
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace {
/*@brief Split a string into a string vector based on the delimiters.
 *
 *@param str String to be splitted.
 *@param delimiters Pointer to the delimiters.
 *@return str_list Substring vector.
 * */
auto string_split(const std::string &str, const char *delimiters)
    -> std::vector<std::string> {
    std::vector<std::string> str_list;
    char *token = std::strtok(const_cast<char *>(str.data()), delimiters);
    while (token != 0) {
        str_list.push_back(token);
        token = std::strtok(nullptr, delimiters);
    }
    return str_list;
}

/*@brief Check if a string contains a substring.
 *
 *@param substr The substring to be checked.
 *@param string The string to be checked.
 *@return bool type.
 * */
bool is_substr(const std::string &substr, const std::string &str) {

    if (str.find(substr) != std::string::npos)
        return true;
    else
        return false;
}
/*@brief Parse the backend details based on the output of
 *Kokkos::print_configuration.
 *
 *@return meta_map std::unordered_map<std::string,
 *std::unordered_map<std::string, std::string>> type.
 * */
auto getConfig() {

    std::ostringstream buffer;

    Kokkos::print_configuration(buffer, true);

    const std::string bufferstr(buffer.str());

    const auto str_list = string_split(bufferstr, "\n");

    std::vector<std::string> query_keys;

    using value_map = std::unordered_map<std::string, std::string>;

    using category_map = std::unordered_map<std::string, value_map>;

    category_map meta_map;

    for (std::size_t i = 0; i < str_list.size(); i++) {

        const std::string tmp_str = str_list[i];

        const auto tmp_str_list = string_split(tmp_str, ":");

        if (i == 0) {
            const auto tmp_str_list1 = string_split(tmp_str_list[1], " ");
            meta_map[tmp_str_list[0]][tmp_str_list[0]] = tmp_str_list1[0];
            query_keys.push_back(tmp_str_list[0]);
        } else {
            // Current string or line only contains a key of category_map.
            if (tmp_str_list.size() == 1) {
                // Append key to the back of the query_keys vector.
                query_keys.push_back(tmp_str_list[0]);
            } else {
                const std::string substr = "KOKKOS_ENABLE";
                if (is_substr(substr, tmp_str_list[0])) { // remove space
                    const auto tmp_str_list0 =
                        string_split(tmp_str_list[0], " ");

                    const auto tmp_str_list1 =
                        string_split(tmp_str_list[1], " ");
                    meta_map[query_keys.back()][tmp_str_list0[0]] =
                        tmp_str_list1[0];
                } else {
                    const auto tmp_str_list1 =
                        string_split(tmp_str_list[1], " ");

                    meta_map[query_keys.back()][tmp_str_list[0]] =
                        tmp_str_list1[0];
                }
            }
        }
    }

    auto backend_str = meta_map["Architecture"]["Default Device"];

    if (is_substr("Serial", backend_str)) {
        meta_map["Backend"]["Serial"] = "yes";
    } else if (is_substr("OpenMP", backend_str)) {
        meta_map["Backend"]["Parallel"] = "OpenMP";
    } else if (is_substr("Cuda", backend_str)) {
        meta_map["Backend"]["Parallel"] = "CUDA";
    } else {
        meta_map["Backend"]["Parallel"] = "HIP";
    }

    return meta_map;
}
} // namespace
