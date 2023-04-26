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
    using value_map = std::unordered_map<std::string, std::string>;
    using category_map = std::unordered_map<std::string, value_map>;

    std::ostringstream buffer;
    Kokkos::print_configuration(buffer, true);
    const std::string bufferstr(buffer.str());
    const auto str_list = string_split(bufferstr, "\n");

    std::vector<std::string> query_keys;
    std::size_t num_devices = 0;
    category_map meta_map;

    // Kokkos version
    const std::string tmp_str0 = str_list[0];
    const auto tmp_str_list0 = string_split(tmp_str0, ":");
    const auto tmp_str_list01 = string_split(tmp_str_list0[1], " ");
    meta_map[tmp_str_list0[0]][tmp_str_list0[0]] = tmp_str_list01[0];
    query_keys.push_back(tmp_str_list0[0]);

    for (std::size_t i = 1; i < str_list.size(); i++) {
        const std::string tmp_str = str_list[i];
        // If last char of string is ':', this string is key.
        if (str_list[i].back() == ':') {
            // Append key to the back of the query_keys vector.
            query_keys.push_back(tmp_str.substr(0, tmp_str.size() - 1));
            if (query_keys.back() == "Serial Runtime Configuration") {
                meta_map[query_keys.back()]["Serial"] = "yes";
            }
        } else {
            if (query_keys.size() > 7) {

                if (query_keys.back() == "OpenMP Runtime Configuration") {
                    meta_map[query_keys.back()]["OpenMP"] = tmp_str;
                } else if (query_keys.back() == "Cuda Runtime Configuration" ||
                           query_keys.back() == "Runtime Configuration") {
                    std::string backend = meta_map["Backend"]["Parallel"];
                    if (is_substr("KOKKOS_ENABLE_" + backend, tmp_str)) {
                        meta_map[query_keys.back()]
                                ["KOKKOS_ENABLE_" + backend] = "defined";
                    } else if (is_substr(backend + "_VERSION", tmp_str)) {
                        const auto tmp_str_list_equal =
                            string_split(tmp_str, "=");
                        meta_map[query_keys.back()][backend + "_VERSION"] =
                            string_split(tmp_str_list_equal.back(), " ").back();
                    } else {
                        std::string device_id = std::to_string(num_devices);
                        device_id.append("th device");
                        meta_map[query_keys.back()][device_id] = tmp_str;
                        num_devices++;
                    }
                }
            } else {
                const auto tmp_str_list = string_split(tmp_str, ":");
                const auto tmp_str_list1 = string_split(tmp_str_list[1], " ");
                meta_map[query_keys.back()][tmp_str_list[0]] = tmp_str_list1[0];

                if (query_keys.back() == "Architecture") {
                    const std::string prefix_sub = "NXKokkosX";
                    std::string sub = tmp_str_list1[0].substr(
                        prefix_sub.size(),
                        tmp_str_list1[0].size() - prefix_sub.size() - 1);
                    if (sub == "Serial") {
                        meta_map["Backend"]["Serial"] = "yes";
                    } else {
                        std::transform(sub.begin(), sub.end(), sub.begin(),
                                       ::toupper);
                        meta_map["Backend"]["Parallel"] = sub;
                    }
                }
            }
        }
    }

    return meta_map;
}
} // namespace
