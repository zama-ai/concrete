cmake_minimum_required(VERSION 3.22)
project(damgard)

set(CMAKE_CXX_STANDARD 17)
include(FindPackageHandleStandardArgs)

# Find HCS
find_path(HCS_INCLUDE_DIR libhcs++.hpp)
find_library(HCS_LIBRARY NAMES hcs)
find_package_handle_standard_args(HCS DEFAULT_MSG HCS_INCLUDE_DIR HCS_LIBRARY)
mark_as_advanced(HCS_INCLUDE_DIR HCS_LIBRARY)
if(HCS_FOUND)
    message("HCS found")
    include_directories(${HCS_INCLUDE_DIR})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${HCS_LIBRARY})
else()
    message("HCS not found")
endif()

# Find GMP
find_path(GMP_INCLUDE_DIR gmp.h)
find_library(GMP_LIBRARY NAMES gmp)
find_package_handle_standard_args(GMP DEFAULT_MSG GMP_INCLUDE_DIR GMP_LIBRARY)
mark_as_advanced(GMP_INCLUDE_DIR GMP_LIBRARY)
if(GMP_FOUND)
    message("GMP found")
    include_directories(${GMP_INCLUDE_DIR})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${GMP_LIBRARY})
else()
    message("GMP not found")
endif()

find_package(OpenMP)
# Your source files
set(SOURCE_FILES defines.h library.cpp library.h utils.cpp utils.h)
add_executable(compresslwe main.cpp ${SOURCE_FILES})
add_executable(tests tests.cpp ${SOURCE_FILES})
# Link your application with the HCS library
add_executable(damgard damgard.cpp ${SOURCE_FILES})
target_link_libraries(damgard ${EXTRA_LIBS})
target_link_libraries(compresslwe ${EXTRA_LIBS})
target_link_libraries(tests ${EXTRA_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(compresslwe
            # OpenMP::OpenMP_CXX is an imported target with INTERFACE_**
            $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> )
    target_link_libraries(damgard
            # OpenMP::OpenMP_CXX is an imported target with INTERFACE_**
            $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> )
    target_link_libraries(tests
            # OpenMP::OpenMP_CXX is an imported target with INTERFACE_**
            $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> )
endif()