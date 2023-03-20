project(lightning_kokkos_algorithms LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)


set(Kokkos_ALGORITHM_FILES  AdjointDiffKokkos.hpp 
                            AdjointDiffKokkos.cpp
                            CACHE INTERNAL "" FORCE)
add_library(lightning_kokkos_algorithms STATIC ${Kokkos_ALGORITHM_FILES})

target_include_directories(lightning_kokkos_algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(lightning_kokkos_algorithms PUBLIC lightning_kokkos_simulator lightning_kokkos_utils) 
set_property(TARGET lightning_kokkos_algorithms PROPERTY POSITION_INDEPENDENT_CODE ON)

