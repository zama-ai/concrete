# concrete-ffi/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

project(Concrete)

set(CONCRETE_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../target/release)
include_directories(${CONCRETE_BUILD_DIR})
link_directories(${CONCRETE_BUILD_DIR})

set(SOURCE tests/tests.c)

add_executable(tests ${SOURCE})
target_link_libraries (tests LINK_PUBLIC concrete_ffi)
