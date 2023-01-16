cmake_minimum_required(VERSION 3.17)

project(concretecompiler LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Needed on linux with clang 15 and on MacOS because cxx emits dollars in the optimizer C++ API
add_definitions("-Wno-dollar-in-identifier-extension")

add_definitions("-Wall ")
add_definitions("-Werror ")

# If we are trying to build the compiler with LLVM/MLIR as libraries
if(NOT DEFINED LLVM_EXTERNAL_CONCRETELANG_SOURCE_DIR)
  message(FATAL_ERROR "Concrete compiler requires a unified build with LLVM/MLIR")
endif()

# CMake library generation settings.
set(BUILD_SHARED_LIBS
    OFF
    CACHE BOOL "Default to building a static mondo-lib")
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME
    ON
    CACHE BOOL "Python soname linked libraries are bad")
set(CMAKE_VISIBILITY_INLINES_HIDDEN
    ON
    CACHE BOOL "Hide inlines")

# The -fvisibility=hidden option only works for static builds.
if(BUILD_SHARED_LIBS AND (CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden"))
  message(FATAL_ERROR "CMAKE_CXX_VISIBILITY_PRESET=hidden is incompatible \
                        with BUILD_SHARED_LIBS.")
endif()

set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir) # --src-root
set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include) # --includedir
set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Custom doc generation function
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddConcretelangDoc)
set(CONCRETELANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# -------------------------------------------------------------------------------
# Concrete Security curves Configuration
# -------------------------------------------------------------------------------
include_directories(${PROJECT_SOURCE_DIR}/parameter-curves/concrete-security-curves-cpp/include)

# -------------------------------------------------------------------------------
# Concrete FFI Configuration
# -------------------------------------------------------------------------------
include_directories(${CONCRETE_FFI_RELEASE})
add_library(Concrete STATIC IMPORTED)
set_target_properties(Concrete PROPERTIES IMPORTED_LOCATION ${CONCRETE_FFI_RELEASE}/libconcrete_core_ffi.a)

# --------------------------------------------------------------------------------
# Concrete Cuda Configuration
# --------------------------------------------------------------------------------
option(CONCRETELANG_CUDA_SUPPORT "Support Concrete CUDA Execution." OFF)

if(CONCRETELANG_CUDA_SUPPORT)
  if(NOT DEFINED CONCRETE_CORE_PATH)
    message(FATAL_ERROR "Compiling with CUDA support requires setting CONCRETE_CORE_PATH")
  endif()
  remove_definitions("-Werror ")
  message(STATUS "Building with Concrete CUDA execution support")
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Found CUDA version: ${CUDAToolkit_VERSION}")
  message(STATUS "Found CUDA library dir: ${CUDAToolkit_LIBRARY_DIR}")
  link_directories(${CUDAToolkit_LIBRARY_DIR})
  add_subdirectory(${CONCRETE_CORE_PATH}/concrete-cuda/cuda)
  include_directories(${CONCRETE_CORE_PATH}/concrete-cuda/cuda/include)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  add_compile_options(-DCONCRETELANG_CUDA_SUPPORT)
endif()

# --------------------------------------------------------------------------------
# Python Configuration
# -------------------------------------------------------------------------------
option(CONCRETELANG_BINDINGS_PYTHON_ENABLED "Enables ConcreteLang Python bindings." ON)

if(CONCRETELANG_BINDINGS_PYTHON_ENABLED)
  message(STATUS "ConcreteLang Python bindings are enabled.")

  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
  set(CONCRETELANG_PYTHON_PACKAGES_DIR ${CMAKE_CURRENT_BINARY_DIR}/python_packages)
else()
  message(STATUS "ConcreteLang Python bindings are disabled.")
endif()

# -------------------------------------------------------------------------------
# DFR - parallel execution configuration
# -------------------------------------------------------------------------------
option(CONCRETELANG_DATAFLOW_EXECUTION_ENABLED "Enables dataflow execution for ConcreteLang." ON)
option(CONCRETELANG_TIMING_ENABLED "Enables execution timing." ON)

if(CONCRETELANG_DATAFLOW_EXECUTION_ENABLED)
  message(STATUS "ConcreteLang dataflow execution enabled.")

  find_package(HPX REQUIRED CONFIG)
  include_directories(SYSTEM ${HPX_INCLUDE_DIRS})
  list(APPEND CMAKE_MODULE_PATH "${HPX_CMAKE_DIR}")
  add_compile_options(-DCONCRETELANG_DATAFLOW_EXECUTION_ENABLED
                      -DHPX_DEFAULT_CONFIG_FILE="${PROJECT_SOURCE_DIR}/hpx.ini")

else()
  message(STATUS "ConcreteLang dataflow execution disabled.")
endif()

if(CONCRETELANG_TIMING_ENABLED)
  add_compile_options(-DCONCRETELANG_TIMING_ENABLED)
else()
  message(STATUS "ConcreteLang execution timing disabled.")
endif()

# -------------------------------------------------------------------------------
# Unit tests
# -------------------------------------------------------------------------------
option(CONCRETELANG_UNIT_TESTS "Enables the build of unittests" ON)

# -------------------------------------------------------------------------------
# Benchmarks
# -------------------------------------------------------------------------------
option(CONCRETELANG_BENCHMARK "Enables the build of benchmarks" ON)

# -------------------------------------------------------------------------------
# Handling sub dirs
# -------------------------------------------------------------------------------
include_directories(${CONCRETE_OPTIMIZER_DIR}/concrete-optimizer-cpp/src/cpp)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(tests)

add_subdirectory(${CONCRETE_OPTIMIZER_DIR}/concrete-optimizer-cpp/cmake-utils)
