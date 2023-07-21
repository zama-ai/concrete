cmake_minimum_required(VERSION 3.22)
project(damgard)

set(CMAKE_CXX_STANDARD 17)
include(FindPackageHandleStandardArgs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(libhcs)
include_directories(libhcs/include)
include_directories(include)

set(CMAKE_BUILD_TYPE Release)
set(EXTRA_LIBS ${EXTRA_LIBS} hcs omp)

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

if(APPLE)
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)

  if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
  endif()
endif()

find_package(OpenMP REQUIRED)

add_library(compresslwe STATIC src/defines.cpp src/library.cpp src/utils.cpp src/serialize.cpp)

# set_source_files_properties(library.cpp PROPERTIES COMPILE_FLAGS "-fopenmp")

add_executable(main src/main.cpp)
add_executable(tests src/tests.cpp)
# Link your application with the HCS library
add_executable(damgard src/damgard.cpp)
target_link_libraries(damgard ${EXTRA_LIBS})
target_link_libraries(main ${EXTRA_LIBS})
target_link_libraries(tests ${EXTRA_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(main
            # OpenMP::OpenMP_CXX is an imported target with INTERFACE_**
            $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> compresslwe hcs)
    target_link_libraries(damgard
            # OpenMP::OpenMP_CXX is an imported target with INTERFACE_**
            $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> compresslwe hcs)
    target_link_libraries(tests
            # OpenMP::OpenMP_CXX is an imported target with INTERFACE_**
            $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> compresslwe hcs)
endif()

install(TARGETS compresslwe EXPORT compresslwe)
install(EXPORT compresslwe DESTINATION ${CMAKE_INSTALL_PREFIX})
