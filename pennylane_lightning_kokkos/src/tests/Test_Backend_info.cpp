#include "Bindings.hpp"
#include "Error.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"
#include <algorithm>
#include <cassert>
#include <catch2/catch.hpp>
#include <cctype>

using namespace Pennylane;
namespace {} // namespace

TEMPLATE_TEST_CASE("Bindings::getConfig", "[Backend Info]", float) {
    const std::size_t num_qubits = 3;
    StateVectorKokkos<TestType> kokkos_sv{num_qubits};

    SECTION("Check string split") {
        const std::string str = "Check string split!";
        const auto str_list = string_split(str, " ");
        PL_ASSERT(str_list[0] == "Check");
        PL_ASSERT(str_list[1] == "string");
        PL_ASSERT(str_list[2] == "split!")
    }

    SECTION("Check string contain") {
        const std::string str = "Check string contain!";
        const std::string substr0 = "Check";
        bool issubstr = is_substr(substr0, str);
        PL_ASSERT(issubstr == true);

        const std::string substr1 = "CHECK";
        issubstr = is_substr(substr1, str);
        PL_ASSERT(issubstr == false);

        const std::vector<std::string> sub_str_vec0 = {"contain", "List",
                                                       "Test"};
        bool contain_substr = is_substr(sub_str_vec0, str);
        PL_ASSERT(contain_substr == true);

        const std::vector<std::string> sub_str_vec1 = {"Unit", "List", "Test"};
        contain_substr = is_substr(sub_str_vec1, str);
        PL_ASSERT(contain_substr == false);
    }

    SECTION("Check All Info") {
        auto config_info = getConfig();

        std::vector<std::string> query_categories = {
            "Kokkos Version", "Compiler",      "Arch",
            "Atomics",        "Vectorization", "Memory",
            "Options",        "Backend",       "Runtime Config",
        };

        std::unordered_map<std::string, std::vector<std::string>> query_keys = {
            {"Kokkos Version", {"Kokkos Version"}},
            {"Compiler",
             {"KOKKOS_COMPILER_APPLECC", "KOKKOS_COMPILER_CLANG",
              "KOKKOS_COMPILER_CRAYC", "KOKKOS_COMPILER_GNU",
              "KOKKOS_COMPILER_IBM", "KOKKOS_COMPILER_INTEL",
              "KOKKOS_COMPILER_NVCC", "KOKKOS_COMPILER_PGI",
              "KOKKOS_COMPILER_MSVC"}},
            {"Atomics",
             {"KOKKOS_ENABLE_GNU_ATOMICS", "KOKKOS_ENABLE_INTEL_ATOMICS",
              "KOKKOS_ENABLE_WINDOWS_ATOMICS"}},
            {"Vectorization",
             {"KOKKOS_ENABLE_PRAGMA_IVDEP", "KOKKOS_ENABLE_PRAGMA_LOOPCOUNT",
              "KOKKOS_ENABLE_PRAGMA_SIMD", "KOKKOS_ENABLE_PRAGMA_UNROLL",
              "KOKKOS_ENABLE_PRAGMA_VECTOR"}},
            {"Memory",
             {"KOKKOS_ENABLE_HBWSPACE", "KOKKOS_ENABLE_INTEL_MM_ALLOC"}},
            {"Options",
             {"KOKKOS_ENABLE_ASM", "KOKKOS_ENABLE_CXX14", "KOKKOS_ENABLE_CXX17",
              "KOKKOS_ENABLE_CXX20", "KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK",
              "KOKKOS_ENABLE_HWLOC", "KOKKOS_ENABLE_LIBRT",
              "KOKKOS_ENABLE_LIBDL"}},
            {"Arch", {"Default Device"}},
            {"Backend", {"Serial", "Parallel"}},
            {"Runtime Config", {"Serial", "Parallel"}}};

        for (auto &category : query_categories) {

            PL_ASSERT(config_info.find(category) != config_info.end());

            bool key_found = false;

            for (auto &key : query_keys[category]) {
                if (config_info[category].find(key) !=
                    config_info[category].end()) {
                    key_found = true;
                }
            }

            PL_ASSERT(key_found == true);

            std::vector<std::string> sub_query_category = {
                "Atomics", "Vectorization", "Memory", "Options"};

            if (category == "Kokkos Version") {
                std::string version_info =
                    config_info["Kokkos Version"]["Kokkos Version"];
                for (auto &c : version_info) {
                    bool is_alpha = isalpha(c);
                    PL_ASSERT(is_alpha == false);
                }
            } else if (category == "Comipler" || category == "Arch") {
                for (auto &key : query_keys[category]) {
                    if (config_info[category].find(key) !=
                        config_info[category].end()) {
                        auto str = config_info[category][key];
                        PL_ASSERT(str.length() != 0);
                    }
                }
            } else if (std::find(sub_query_category.begin(),
                                 sub_query_category.end(),
                                 category) != sub_query_category.end()) {
                for (auto &key : query_keys[category]) {
                    if (config_info[category].find(key) !=
                        config_info[category].end()) {
                        auto str = config_info[category][key];
                        PL_ASSERT(str == "yes" || str == "no");
                    }
                }
            } else if (category == "Backend") {
                for (auto &key : query_keys[category]) {
                    if (config_info[category].find(key) !=
                        config_info[category].end()) {
                        if (key == "Serial") {
                            assert(config_info[category][key] == "Serial");
                        } else if (key == "Parallel") {
                            std::vector<std::string> backend_list = {
                                "OpenMP", "HIP", "CUDA"};
                            bool is_found =
                                std::find(backend_list.begin(),
                                          backend_list.end(),
                                          config_info[category][key]) !=
                                backend_list.end();
                            PL_ASSERT(is_found == true);
                        }
                    }
                }
            } else if (category == "Runtime Config") {
                for (auto &key : query_keys[category]) {
                    if (config_info[category].find(key) !=
                        config_info[category].end()) {
                        if (key == "Serial") {
                            PL_ASSERT(config_info[category][key] == "Serial");
                        } else if (key == "Parallel") {
                            auto str = config_info[category][key];
                            PL_ASSERT(str.length() != 0);
                        }
                    }
                }
            }
        }
    } // End of Section
}
