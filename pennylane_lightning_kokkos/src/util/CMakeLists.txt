project(lightning_kokkos_utils LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(UTIL_FILES Error.hpp Util.hpp CACHE INTERNAL "" FORCE)

add_library(lightning_kokkos_utils INTERFACE ${UTIL_FILES})
target_include_directories(lightning_kokkos_utils INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/DS>
)