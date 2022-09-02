project(lightning_kokkos_components LANGUAGES CXX)

###############################################################################
# Include all nested sources directories
###############################################################################
set(COMPONENT_SUBDIRS
    simulator;
    util;
    algorithms;
)
foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()

if (PLKOKKOS_BUILD_TESTS)
    add_subdirectory("tests" "tests")
endif()
