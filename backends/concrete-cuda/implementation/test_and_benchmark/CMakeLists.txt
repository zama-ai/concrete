option(CONCRETE_CUDA_BUILD_TESTS "Build the test tool" ON)
option(CONCRETE_CUDA_BUILD_BENCHMARKS "Build the benchmark tool" ON)

if(CONCRETE_CUDA_BUILD_TESTS OR CONCRETE_CUDA_BUILD_BENCHMARKS)
  set(CONCRETE_CPU_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../concrete-cpu/implementation/target/release")
  set(CONCRETE_CPU_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../concrete-cpu/implementation")

  if(NOT TARGET concrete_cpu)
    # Enable ExternalProject CMake module
    include(ExternalProject)

    # Add rust_example as a CMake target
    ExternalProject_Add(
      concrete_cpu
      SOURCE_DIR ${CONCRETE_CPU_SOURCE_DIR}
      DOWNLOAD_COMMAND ""
      CONFIGURE_COMMAND ""
      BUILD_COMMAND cargo +nightly build --release --features=nightly
      BINARY_DIR ${CONCRETE_CPU_BINARY_DIR}
      BUILD_ALWAYS true
      INSTALL_COMMAND ""
      LOG_BUILD ON)
  endif()
endif()

if(CONCRETE_CUDA_BUILD_TESTS)
  message(STATUS "Building with Concrete CUDA test tool")
  add_subdirectory(test)
endif()
if(CONCRETE_CUDA_BUILD_BENCHMARKS)
  message(STATUS "Building with Concrete CUDA benchmark tool")
  add_subdirectory(benchmark)
endif()
