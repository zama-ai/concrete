find_package(CUDAToolkit)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Disable the Google Benchmark requirement on Google Test
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_TESTING OFF)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.7.1)
FetchContent_MakeAvailable(googlebenchmark)

set(CONCRETE_CUDA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")
set(TFHE_RS_BINARY_DIR "${CMAKE_BINARY_DIR}/test_and_benchmark/tfhe_rs-prefix/src/tfhe_rs/target/release")
set(TFHE_RS_SOURCE_DIR "${CMAKE_BINARY_DIR}/test_and_benchmark/tfhe_rs-prefix/src/tfhe_rs")
if(NOT TARGET tfhe_rs)
  include(ExternalProject)

  ExternalProject_Add(
          tfhe_rs
          GIT_REPOSITORY https://github.com/zama-ai/tfhe-rs.git
          GIT_TAG origin/main
          STEP_TARGETS build
          SOURCE_DIR ${TFHE_RS_SOURCE_DIR}
          CONFIGURE_COMMAND ""
          BUILD_COMMAND make build_c_api -C <SOURCE_DIR>
          BUILD_ALWAYS true
          INSTALL_COMMAND ""
          LOG_BUILD ON)
endif()


add_library(concrete_cpu_lib STATIC IMPORTED)
set_target_properties(concrete_cpu_lib PROPERTIES IMPORTED_LOCATION ${CONCRETE_CPU_BINARY_DIR}/libconcrete_cpu.a)

add_library(tfhe_rs_lib SHARED IMPORTED)
set_target_properties(tfhe_rs_lib PROPERTIES IMPORTED_LOCATION ${TFHE_RS_BINARY_DIR}/libtfhe.so)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CONCRETE_CUDA_SOURCE_DIR}/include)
include_directories(${CONCRETE_CPU_SOURCE_DIR}/include)
include_directories(${TFHE_RS_BINARY_DIR})

find_package(OpenMP REQUIRED)
# Add the OpenMP flag to the compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl")
set(BINARY benchmark_concrete_cuda)

file(
  GLOB_RECURSE BENCH_SOURCES
  LIST_DIRECTORIES false
  benchmark*.cpp main.cpp)
set(SOURCES ${BENCH_SOURCES})

add_executable(${BINARY} ${BENCH_SOURCES} ../utils.cpp ../setup_and_teardown.cpp)

set_target_properties(benchmark_concrete_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_dependencies(benchmark_concrete_cuda tfhe_rs)
target_link_libraries(
  benchmark_concrete_cuda
  PUBLIC benchmark::benchmark concrete_cpu_lib tfhe_rs_lib concrete_cuda OpenMP::OpenMP_CXX
  PRIVATE CUDA::cudart)
