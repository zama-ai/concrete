find_package(CUDAToolkit)
include(FetchContent)
FetchContent_Declare(googletest
                     URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl")

set(BINARY test_concrete_cuda)

file(
  GLOB_RECURSE TEST_SOURCES
  LIST_DIRECTORIES false
  *.h *.cpp)

set(SOURCES ${TEST_SOURCES})

add_executable(${BINARY} ${TEST_SOURCES} ../utils.cpp ../setup_and_teardown.cpp)

add_test(NAME ${BINARY} COMMAND ${BINARY})

set_target_properties(
  test_concrete_cuda
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
             CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CUDA_ARCHITECTURES native)
add_dependencies(test_concrete_cuda tfhe_rs)
target_link_libraries(test_concrete_cuda PUBLIC GTest::gtest_main concrete_cpu_lib tfhe_rs_lib concrete_cuda cudart)
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}" "${CMAKE_CURRENT_SOURCE_DIR}")

include(GoogleTest)
gtest_discover_tests(test_concrete_cuda)
