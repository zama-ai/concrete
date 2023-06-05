cmake_minimum_required(VERSION 3.21)
project(compresslwe)

set(CMAKE_CXX_STANDARD 17)

find_package(IPCL 2.0.0 REQUIRED)

add_executable(compresslwe main.cpp defines.h library.cpp library.h utils.cpp utils.h)
add_executable(tests tests.cpp defines.h library.cpp library.h utils.cpp utils.h)

target_link_libraries(compresslwe IPCL::ipcl)
target_link_libraries(tests IPCL::ipcl)
