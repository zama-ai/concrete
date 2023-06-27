find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Disable the Google Benchmark requirement on Google Test
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_TESTING OFF)

include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.7.1)
FetchContent_MakeAvailable(googlebenchmark)

# Enable ExternalProject CMake module
include(ExternalProject)

set(CONCRETE_CPU_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../concrete-cpu/implementation/target/release")
set(CONCRETE_CPU_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../concrete-cpu/implementation")
set(CONCRETE_CUDA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

# Enable ExternalProject CMake module
include(ExternalProject)

if(NOT TARGET concrete_cpu)
  ExternalProject_Add(
    concrete_cpu
    SOURCE_DIR ${CONCRETE_CPU_SOURCE_DIR}
    DOWNLOAD_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cargo +nightly build --release --features=nightly
    COMMAND cargo +nightly build --release --features=nightly
    BINARY_DIR ${CONCRETE_CPU_BINARY_DIR}
    BUILD_ALWAYS true
    INSTALL_COMMAND ""
    LOG_BUILD ON)
endif()

set(TFHE_RS_SOURCE_DIR "${CMAKE_BINARY_DIR}/tfhe-rs")
set(TFHE_RS_BINARY_DIR "${TFHE_RS_SOURCE_DIR}/target/release")

if(NOT TARGET tfhe-rs)
  ExternalProject_Add(
    tfhe-rs
    GIT_REPOSITORY https://github.com/zama-ai/tfhe-rs.git
    GIT_TAG main
    SOURCE_DIR ${TFHE_RS_SOURCE_DIR}
    BUILD_IN_SOURCE 1
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make build_c_api
    INSTALL_COMMAND ""
    LOG_BUILD ON)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CONCRETE_CPU_SOURCE_DIR}/include)
include_directories(${CONCRETE_CUDA_SOURCE_DIR}/include)
include_directories(${TFHE_RS_BINARY_DIR})
include_directories("${CUDA_INCLUDE_DIRS}" "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(OpenMP REQUIRED)
# Add the OpenMP flag to the compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_library(concrete_cpu_lib STATIC IMPORTED)
add_dependencies(concrete_cpu_lib concrete_cpu)
set_target_properties(concrete_cpu_lib PROPERTIES IMPORTED_LOCATION ${CONCRETE_CPU_BINARY_DIR}/libconcrete_cpu.a)

add_library(tfhe_rs_lib STATIC IMPORTED)
add_dependencies(tfhe_rs_lib tfhe-rs)
set_target_properties(tfhe_rs_lib PROPERTIES IMPORTED_LOCATION ${TFHE_RS_BINARY_DIR}/libtfhe.a)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed,--allow-multiple-definition -ldl")

set(BINARY benchmark_concrete_cuda)

file(
  GLOB_RECURSE BENCH_SOURCES
  LIST_DIRECTORIES false
  benchmark*.cpp main.cpp)

add_executable(${BINARY} ${BENCH_SOURCES} ../utils.cpp ../setup_and_teardown.cpp)

set_target_properties(benchmark_concrete_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(
  benchmark_concrete_cuda
  PUBLIC benchmark::benchmark concrete_cpu_lib tfhe_rs_lib concrete_cuda OpenMP::OpenMP_CXX
  PRIVATE CUDA::cudart)
