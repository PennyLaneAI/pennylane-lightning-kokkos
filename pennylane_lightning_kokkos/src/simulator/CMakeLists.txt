project(lightning_kokkos_simulator)
set(CMAKE_CXX_STANDARD 17)
enable_language(CXX)

set(SIMULATOR_FILES StateVectorKokkos.cpp
                    StateVectorKokkos.hpp
                    GateFunctors.hpp
                    GatesHost.hpp
                    ExpValFunctors.hpp
                    MeasuresFunctors.hpp
                    MeasuresKokkos.cpp
                    MeasuresKokkos.hpp
                    CACHE INTERNAL "" FORCE)
add_library(lightning_kokkos_simulator STATIC ${SIMULATOR_FILES})
target_include_directories(lightning_kokkos_simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${kokkos_INC_DIR})
target_link_libraries(lightning_kokkos_simulator PUBLIC lightning_kokkos_utils )
target_link_libraries(lightning_kokkos_simulator PUBLIC kokkos)
set_target_properties(lightning_kokkos_simulator PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(lightning_kokkos_simulator PROPERTIES LINKER_LANGUAGE CXX)
if(PLKOKKOS_ENABLE_NATIVE)
    target_compile_options(lightning_kokkos_simulator PRIVATE -march=native)
endif()
