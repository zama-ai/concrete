cmake_minimum_required(VERSION 3.16)

project(Concrete)

set(CONCRETE_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/target)
include_directories(${CONCRETE_BUILD_DIR})
link_directories(${CONCRETE_BUILD_DIR}/release)
add_subdirectory(concrete-npe)
