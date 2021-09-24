cmake_minimum_required(VERSION 3.13)

project(zamacompiler LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# If we are trying to build the compiler with LLVM/MLIR as libraries
if( NOT DEFINED LLVM_EXTERNAL_ZAMALANG_SOURCE_DIR )
  message(FATAL_ERROR "Concrete compiler requires a unified build with LLVM/MLIR")
endif()

# CMake library generation settings.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Default to building a static mondo-lib")
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON CACHE BOOL
  "Python soname linked libraries are bad")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON CACHE BOOL "Hide inlines")

# The -fvisibility=hidden option only works for static builds.
if (BUILD_SHARED_LIBS AND (CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden"))
  message(FATAL_ERROR "CMAKE_CXX_VISIBILITY_PRESET=hidden is incompatible \
                        with BUILD_SHARED_LIBS.")
endif()

set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir ) # --src-root
set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# Concrete FFI Configuration
#-------------------------------------------------------------------------------
include_directories(${CONCRETE_FFI_RELEASE})
add_library(Concrete STATIC IMPORTED)
set_target_properties(Concrete PROPERTIES IMPORTED_LOCATION ${CONCRETE_FFI_RELEASE}/libconcrete_ffi.a )

#-------------------------------------------------------------------------------
# Python Configuration
#-------------------------------------------------------------------------------

option(ZAMALANG_BINDINGS_PYTHON_ENABLED "Enables ZamaLang Python bindings." ON)

if(ZAMALANG_BINDINGS_PYTHON_ENABLED)
  message(STATUS "ZamaLang Python bindings are enabled.")

  include(MLIRDetectPythonEnv)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  message(STATUS "Found Python include dirs: ${Python3_INCLUDE_DIRS}")
  message(STATUS "Found Python libraries: ${Python3_LIBRARIES}")
  message(STATUS "Found Python executable: ${Python3_EXECUTABLE}")

  mlir_detect_pybind11_install()
  find_package(pybind11 2.6 CONFIG REQUIRED)
  message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIR}")
  message(STATUS "Python prefix = '${PYTHON_MODULE_PREFIX}', "
                 "suffix = '${PYTHON_MODULE_SUFFIX}', "
                 "extension = '${PYTHON_MODULE_EXTENSION}'")

  set(ZAMALANG_PYTHON_PACKAGES_DIR ${CMAKE_CURRENT_BINARY_DIR}/python_packages)
else()
  message(STATUS "ZamaLang Python bindings are disabled.")
endif()

#-------------------------------------------------------------------------------
# Unit tests
#-------------------------------------------------------------------------------

option(ZAMALANG_UNIT_TESTS "Enables the build of unittests" ON)


add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(tests)
