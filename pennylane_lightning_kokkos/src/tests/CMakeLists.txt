cmake_minimum_required(VERSION 3.21)

project(pennylane_lightning_kokkos_tests)

set(CMAKE_CXX_STANDARD 17)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(PLKOKKOS_ENABLE_NATIVE "Enable native CPU build tuning" OFF)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.7
)

FetchContent_MakeAvailable(Catch2)

# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# Modify `ctest` to only run the supported subset of tests.
include(CTest)
include(Catch)

add_executable(runner_kokkos runner_main.cpp)

if(PLKOKKOS_ENABLE_OPENMP)
find_package(OpenMP REQUIRED)
target_link_libraries(runner_kokkos INTERFACE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(runner_kokkos PUBLIC pennylane_lightning_kokkos Catch2::Catch2 kokkos)
target_sources(runner_kokkos PRIVATE  Test_LinearAlgebra.cpp
                                      Test_AdjointDiffKokkos.cpp
                                      Test_StateVectorKokkos_Measure.cpp
                                      Test_StateVectorKokkos_NonParam.cpp
                                      Test_StateVectorKokkos_Param.cpp
                                      Test_StateVectorKokkos_Generator.cpp
                                      Test_StateVectorKokkos_Expval.cpp
                                      Test_StateVectorKokkos_Var.cpp)

target_include_directories(runner_kokkos INTERFACE ${kokkos_INC_DIR})
target_compile_options(runner_kokkos PRIVATE "$<$<CONFIG:DEBUG>:-Wall>")

if(PLKOKKOS_ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
    target_compile_options(runner_kokkos PRIVATE -march=native)
endif()

catch_discover_tests(runner_kokkos)
