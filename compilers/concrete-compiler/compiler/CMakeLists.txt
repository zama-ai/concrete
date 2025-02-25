cmake_minimum_required(VERSION 3.17)

project(concretecompiler LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_DEBUG_MODE 0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Needed on linux with clang 15 and on MacOS because cxx emits dollars in the optimizer C++ API
add_definitions("-Wno-dollar-in-identifier-extension")
add_definitions("-Wno-c++98-compat-extra-semi")
add_definitions("-Wno-strict-aliasing")
add_definitions("-Wno-date-time")
add_definitions("-Wno-suggest-override")
add_definitions("-Wno-ctad-maybe-unsupported")
add_definitions("-Wall ")
add_definitions("-Werror ")
add_definitions("-Wfatal-errors")

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

if(DEFINED LLVM_USE_LINKER AND (NOT ${LLVM_USE_LINKER} STREQUAL ""))
  message(INFO " Using custom Linker: ${CMAKE_LINKER}")
else()
  message(INFO " Using standard linker")
endif()

# Custom doc generation function
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddConcretelangDoc)
set(CONCRETELANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# -------------------------------------------------------------------------------
# Concrete Security curves Configuration
# -------------------------------------------------------------------------------
include_directories(${PROJECT_SOURCE_DIR}/../../../tools/parameter-curves/concrete-security-curves-cpp/include)

# -------------------------------------------------------------------------------
# Concrete Protocol
# -------------------------------------------------------------------------------
set(CONCRETE_PROTOCOL_DIR "${PROJECT_SOURCE_DIR}/../../../tools/concrete-protocol")
add_subdirectory(${CONCRETE_PROTOCOL_DIR} concrete-protocol)
get_target_property(CONCRETE_PROTOCOL_GEN_DIR concrete-protocol BINARY_DIR)
set(CONCRETE_PROTOCOL_CAPNP_SRC "${CONCRETE_PROTOCOL_GEN_DIR}/capnp_src_dir/c++/src")
include_directories(${CONCRETE_PROTOCOL_GEN_DIR})
include_directories(${CONCRETE_PROTOCOL_CAPNP_SRC})
add_dependencies(mlir-headers concrete-protocol)
install(TARGETS concrete-protocol EXPORT concrete-protocol)
install(EXPORT concrete-protocol DESTINATION "./")

# -------------------------------------------------------------------------------
# Concrete Optimizer
# -------------------------------------------------------------------------------

set(CONCRETE_OPTIMIZER_DIR "${PROJECT_SOURCE_DIR}/../../concrete-optimizer")
set(CONCRETE_OPTIMIZER_INCLUDE_DIR "${CONCRETE_OPTIMIZER_DIR}/concrete-optimizer-cpp/src/cpp")
include_directories(${CONCRETE_OPTIMIZER_INCLUDE_DIR})

# -------------------------------------------------------------------------------
# Concrete Backends
# -------------------------------------------------------------------------------

set(CONCRETE_BACKENDS_DIR "${PROJECT_SOURCE_DIR}/../../../backends")

# -------------------------------------------------------------------------------
# Concrete CPU Configuration
# -------------------------------------------------------------------------------

set(CONCRETE_CPU_DIR "${CONCRETE_BACKENDS_DIR}/concrete-cpu/implementation")
set(CONCRETE_CPU_INCLUDE_DIR "${CONCRETE_CPU_DIR}/include")

# -------------------------------------------------------------------------------
# Concrete CPU Noise Model Configuration
# -------------------------------------------------------------------------------

set(CONCRETE_CPU_NOISE_MODEL_DIR "${CONCRETE_BACKENDS_DIR}/concrete-cpu/noise-model")
set(CONCRETE_CPU_NOISE_MODEL_INCLUDE_DIR "${CONCRETE_CPU_NOISE_MODEL_DIR}/include")

# -------------------------------------------------------------------------------
# RUST DEPS BUNDLE
# -------------------------------------------------------------------------------

set(RUST_DEPS_BUNDLE_DIR "${PROJECT_SOURCE_DIR}/../../../tools/rust_deps_bundle")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(RUST_DEPS_BUNDLE_BUILD_DIR "${RUST_DEPS_BUNDLE_DIR}/target/debug")
  set(RUST_DEPS_BUNDLE_TARGET "")
else()
  set(RUST_DEPS_BUNDLE_BUILD_DIR "${RUST_DEPS_BUNDLE_DIR}/target/release")
  set(RUST_DEPS_BUNDLE_TARGET "--release")
endif()
set(RUST_DEPS_BUNDLE_LIB "${RUST_DEPS_BUNDLE_BUILD_DIR}/librust_deps_bundle.a")

ExternalProject_Add(
  rust_deps_bundle_build
  DOWNLOAD_COMMAND ""
  CONFIGURE_COMMAND "" OUTPUT "${RUST_DEPS_BUNDLE_LIB}"
  BUILD_ALWAYS true
  BUILD_COMMAND cargo +nightly-2024-09-30 build ${RUST_DEPS_BUNDLE_TARGET}
  BINARY_DIR "${RUST_DEPS_BUNDLE_DIR}"
  INSTALL_COMMAND cp ${RUST_DEPS_BUNDLE_LIB} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  LOG_BUILD ON
  LOG_OUTPUT_ON_FAILURE ON)

add_library(rust_deps_bundle STATIC IMPORTED)
set_target_properties(rust_deps_bundle PROPERTIES IMPORTED_LOCATION "${RUST_DEPS_BUNDLE_LIB}")
add_dependencies(rust_deps_bundle rust_deps_bundle_build)
target_link_libraries(rust_deps_bundle INTERFACE dl)

# --------------------------------------------------------------------------------
# Concrete Cuda Configuration
# --------------------------------------------------------------------------------
option(CONCRETELANG_CUDA_SUPPORT "Support Concrete CUDA Execution." OFF)

set(CONCRETE_CUDA_DIR "${CONCRETE_BACKENDS_DIR}/concrete-cuda/implementation")

if(CONCRETELANG_CUDA_SUPPORT)
  remove_definitions("-Werror ")
  message(STATUS "Building with Concrete CUDA execution support")
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Found CUDA version: ${CUDAToolkit_VERSION}")
  message(STATUS "Found CUDA library dir: ${CUDAToolkit_LIBRARY_DIR}")
  link_directories(${CUDAToolkit_LIBRARY_DIR})
  set(CONCRETE_CUDA_BUILD_TESTS OFF)
  set(CONCRETE_CUDA_BUILD_BENCHMARKS OFF)
  add_subdirectory(${CONCRETE_CUDA_DIR} concrete-cuda)
  include_directories(${CONCRETE_CUDA_DIR}/include)
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

  include(FetchHpxLibrary)
  fetch_hpx_library()

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
option(
  CONCRETELANG_UNSECURE_DEBUG
  "Totally unsecure mode where secret keys are filled with zeros. Useful to reveal the body of the ciphertexts when using tracing during debug."
  OFF)
if(CONCRETELANG_UNSECURE_DEBUG)
  message(
    WARNING
      "
  #############################################################################
  ##                                                                         ##
  ##                            !!! WARNING !!!                              ##
  ##                                                                         ##
  ##  CONCRETELANG_UNSECURE_DEBUG option activated ! This means that the     ##
  ##  secret keys generated will provide ZERO security. This should only be  ##
  ##  used for debugging purpose, and NEVER in a production environment.     ##
  ##                                                                         ##
  #############################################################################
  ")
  add_compile_definitions(CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS)
  add_compile_definitions(CONCRETELANG_TEST_KEYCACHE_PATH="UnsecureKeySetCache")
endif()

# -------------------------------------------------------------------------------
# Benchmarks
# -------------------------------------------------------------------------------
option(CONCRETELANG_BENCHMARK "Enables the build of benchmarks" ON)

# -------------------------------------------------------------------------------
# Handling sub dirs
# -------------------------------------------------------------------------------

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(tests)
