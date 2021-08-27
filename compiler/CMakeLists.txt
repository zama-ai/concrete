cmake_minimum_required(VERSION 3.13)

project(zamacompiler LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIR cmake file from: ${MLIR_DIR}")

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVM cmake file from: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# Concrete FFI Configuration
#-------------------------------------------------------------------------------
include_directories(${CONCRETE_FFI_RELEASE})
add_library(Concrete SHARED IMPORTED)
set_target_properties(Concrete PROPERTIES IMPORTED_LOCATION ${CONCRETE_FFI_RELEASE}/libconcrete_ffi.so )

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

if (ZAMALANG_BINDINGS_PYTHON_ENABLED)
    add_subdirectory(python)
endif()
