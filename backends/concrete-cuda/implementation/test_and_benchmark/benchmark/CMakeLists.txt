cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
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

# Enable ExternalProject CMake module
include(ExternalProject)

set(CONCRETE_CPU_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../concrete-cpu/implementation/target/release")
set(CONCRETE_CPU_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../concrete-cpu/implementation")
set(CONCRETE_CUDA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CONCRETE_CPU_SOURCE_DIR}/include)
include_directories(${CONCRETE_CUDA_SOURCE_DIR}/include)

add_library(concrete_cpu_lib STATIC IMPORTED)
set_target_properties(concrete_cpu_lib PROPERTIES IMPORTED_LOCATION ${CONCRETE_CPU_BINARY_DIR}/libconcrete_cpu.a)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl")

set(BINARY benchmark_concrete_cuda)

file(
  GLOB_RECURSE BENCH_SOURCES
  LIST_DIRECTORIES false
  benchmark*.cpp main.cpp)
set(SOURCES ${BENCH_SOURCES})

add_executable(${BINARY} ${BENCH_SOURCES} ../utils.cpp ../setup_and_teardown.cpp)

set_target_properties(benchmark_concrete_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(
  benchmark_concrete_cuda
  PUBLIC benchmark::benchmark concrete_cpu_lib concrete_cuda
  PRIVATE CUDA::cudart)
