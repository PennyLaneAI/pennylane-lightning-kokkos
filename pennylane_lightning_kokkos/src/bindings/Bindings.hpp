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
#include "Kokkos_Core.hpp"
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace Pennylane {
/*@brief Split a string into a string vector based on the delimiters.
 *
 *@param str String to be splitted.
 *@param delimiters Poniter to the delimiters.
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
/*@brief Check if a string contains substrings that is in a string vector.
 *
 *@param substr The substring vector to be checked.
 *@param string The string to be checked.
 *@return bool type.
 * */
bool is_substr(const std::vector<std::string> &substrs,
               const std::string &str) {
    for (std::size_t i = 0; i < substrs.size(); i++) {
        if (str.find(substrs[i]) != std::string::npos)
            return true;
    }
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

    const std::vector<std::string> query_keys{
        "Kokkos Version", "Compiler",      "Arch",
        "Atomics",        "Vectorization", "Memory",
        "Options",        "Backend",       "Runtime Config"};

    using value_map = std::unordered_map<std::string, std::string>;

    using category_map = std::unordered_map<std::string, value_map>;

    category_map meta_map;

    std::size_t looped_len = 0;

    for (std::size_t i = 0; i < str_list.size(); i++) {
        const std::string tmp_str = str_list[i];
        bool is_key_contained = is_substr(query_keys[looped_len], tmp_str);
        if (looped_len == 0 && is_key_contained) {
            const auto tmp_str_list = string_split(tmp_str, ":");
            const auto tmp_str_list1 = string_split(tmp_str_list[1], " ");
            meta_map[query_keys[looped_len]][tmp_str_list[0]] =
                tmp_str_list1[0];
            looped_len++;
        } else if (looped_len < 7) {
            if (is_key_contained) {
                looped_len++;
            } else {
                const auto tmp_str_list = string_split(tmp_str, ":");
                const auto tmp_str_list1 = string_split(tmp_str_list[1], " ");
                meta_map[query_keys[looped_len - 1]][tmp_str_list[0]] =
                    tmp_str_list1[0];
            }
        } else if (looped_len == 7) {
            std::vector<std::string> substrs{"ASM",   "CXX",   "DEBUG",
                                             "HWLOC", "LIBDL", "LIBRT"};

            if (is_substr(substrs, tmp_str)) {
                const auto tmp_str_list = string_split(tmp_str, ": ");
                meta_map[query_keys[looped_len - 1]][tmp_str_list[0]] =
                    tmp_str_list[1];
            } else if (is_substr(query_keys[looped_len + 1], tmp_str)) {
                std::string substr = "Serial";
                if (is_substr(substr, tmp_str)) {
                    meta_map[query_keys[looped_len]]["Serial"] = "Serial";
                    meta_map[query_keys[looped_len + 1]]["Serial"] = "Serial";
                } else {
                    const auto tmp_str_list0 = string_split(tmp_str, " ");
                    meta_map[query_keys[looped_len]]["Parallel"] =
                        tmp_str_list0[0];
                    meta_map[query_keys[looped_len + 1]]["Parallel"] =
                        str_list[i + 1];
                }
            }
        }
    }
    return meta_map;
}
} // namespace Pennylane
