##########################
## Set Project version
##########################
cmake_minimum_required(VERSION 3.20)
set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░█░█░█▀█░█░█░█░█░█▀█░█▀▀
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░█▀▄░█░█░█▀▄░█▀▄░█░█░▀▀█
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░▀░▀░▀▀▀░▀░▀░▀░▀░▀▀▀░▀▀▀
]=])
message(${LOGO})

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version")

project(pennylane_lightning_kokkos
    DESCRIPTION "Lightning-Kokkos bindings for PennyLane. Backed by the Kokkos library."
    LANGUAGES CXX C
)

##########################
## Utility methods
##########################

# Read and set pennylane_lightning_kokkos version
function(set_pennylane_lightning_kokkos_version VERSION_FILE_PATH)
    file(STRINGS ${VERSION_FILE_PATH} VERSION_FILE_STR)
    foreach (LINE IN LISTS VERSION_FILE_STR)
        if("${LINE}" MATCHES "__version__.*")
            set(VERSION_LINE_STR "${LINE}")
        endif()
    endforeach()
    string(REGEX REPLACE "__version__ = \"(.*)\"" "\\1" VERSION_STRING ${VERSION_LINE_STR})
    set(VERSION_STRING ${VERSION_STRING} PARENT_SCOPE)
endfunction()

set_pennylane_lightning_kokkos_version(${PROJECT_SOURCE_DIR}/pennylane_lightning_kokkos/_version.py)

message(STATUS "pennylane_lightning_kokkos version ${VERSION_STRING}")
set(PROJECT_VERSION ${VERSION_STRING})

##########################
## Enforce C++ Standard
##########################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##########################
## Set Default Options
##########################
# Compiler options
option(PLKOKKOS_ENABLE_NATIVE "Enable native CPU build tuning" OFF)
option(PLKOKKOS_BUILD_TESTS "Build cpp tests" OFF)
option(PLKOKKOS_ENABLE_WARNINGS "Enable warnings" ON)
option(PLKOKKOS_ENABLE_CLANG_TIDY "Enable clang-tidy build checks" OFF)
option(PLKOKKOS_ENABLE_PYTHON "Build Python bindings" ON)
option(PLKOKKOS_ENABLE_SANITIZER "Enable address sanitizer" OFF)

# Kokkos options
if(Kokkos_ENABLE_OPENMP)
  set(PLKOKKOS_ENABLE_OPENMP ON)
endif()

if(Kokkos_ENABLE_THREADS)
  set(PLKOKKOS_ENABLE_THREADS ON)
endif()

# Build options
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Ensure the libraries can see additional libs at same level;
# Required for external deps when loading in Python
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)

if(PLKOKKOS_ENABLE_CLANG_TIDY)
    if (NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                            -extra-arg=-std=c++20;
    )
endif()

if(PLKOKKOS_ENABLE_SANITIZER)
    add_compile_options(-fsanitize=address)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_link_options(-fsanitize=address)
    else ()
        add_link_options(-fsanitize=address -static-libasan)
    endif()
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
ENDIF()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
##########################
## Fetch dependencies
##########################

option(Kokkos_ENABLE_DEPRECATION_WARNINGS "Enable Kokkos deprecation warnings" OFF)

include(FetchContent)
FetchContent_Declare(kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        3.7.00
)
FetchContent_MakeAvailable(kokkos)
get_target_property(PLKOKKOS_KOKKOS_INC_DIR kokkos INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(kokkos PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${PLKOKKOS_KOKKOS_INC_DIR}")


############################
## Create libraries
############################
add_subdirectory(pennylane_lightning_kokkos/src)
add_library(pennylane_lightning_kokkos INTERFACE)
target_link_libraries(pennylane_lightning_kokkos INTERFACE lightning_kokkos_utils lightning_kokkos_simulator lightning_kokkos_algorithms kokkos)
target_include_directories(pennylane_lightning_kokkos INTERFACE "pennylane_lightning_kokkos/src")
target_include_directories(pennylane_lightning_kokkos INTERFACE ${PLKOKKOS_KOKKOS_INC_DIR} )

# Create binding module
if(PLKOKKOS_ENABLE_PYTHON)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.10.1
)
FetchContent_MakeAvailable(pybind11)

find_package (Python COMPONENTS Interpreter Development)
    if(Kokkos_ENABLE_CUDA)
        add_library(lightning_kokkos_qubit_ops MODULE "pennylane_lightning_kokkos/src/bindings/Bindings.cpp")
        target_link_libraries(lightning_kokkos_qubit_ops PUBLIC pybind11::module)
        pybind11_extension(lightning_kokkos_qubit_ops)
    else()
        pybind11_add_module(lightning_kokkos_qubit_ops "pennylane_lightning_kokkos/src/bindings/Bindings.cpp" )
    endif()
    target_link_libraries(lightning_kokkos_qubit_ops PRIVATE pennylane_lightning_kokkos)
    set_target_properties(lightning_kokkos_qubit_ops PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")
    target_compile_options(lightning_kokkos_qubit_ops PRIVATE "$<$<CONFIG:RELEASE>:-W>")
    target_include_directories(lightning_kokkos_qubit_ops PRIVATE ${PLKOKKOS_KOKKOS_INC_DIR})
    target_compile_definitions(lightning_kokkos_qubit_ops PRIVATE VERSION_INFO=${VERSION_STRING})
    target_link_libraries(lightning_kokkos_qubit_ops PRIVATE kokkos)
endif()


if(PLKOKKOS_ENABLE_WARNINGS)
    target_compile_options(pennylane_lightning_kokkos INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror;>
    )
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(pennylane_lightning_kokkos INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=volatile;-Wno-error=deprecated-declarations>
        )
    endif()
    # Explicitly disable warnings for Kokkos-specific library issues.
endif()

if(PLKOKKOS_ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Using -march=native")
    add_compile_options(-march=native)
    target_compile_options(pennylane_lightning_kokkos INTERFACE -march=native)
    target_compile_options(lightning_kokkos_qubit_ops PRIVATE -march=native)
endif()

if (PLKOKKOS_BUILD_TESTS)
    enable_testing()
endif()

