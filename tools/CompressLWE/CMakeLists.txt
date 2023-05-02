cmake_minimum_required(VERSION 3.21)
project(compresslwe)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_BUILD_TYPE Release)

set(IPCL_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/ipcl_bin_dir)
set(IPCL_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/ipcl_install_dir)

include(ExternalProject)

include(ProcessorCount)
ProcessorCount(N_CORES)
if(NOT DEFINED N_CORES OR N_CORES EQUAL 0)
  set(N_CORES 8)
endif()

ExternalProject_Add(ipcl 
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/../../third_party/pailliercryptolib
    BINARY_DIR ${IPCL_BIN_DIR}
    INSTALL_DIR ${IPCL_INSTALL_DIR}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DIPCL_TEST=OFF -DIPCL_BENCHMARK=OFF -DCMAKE_INSTALL_PREFIX=${IPCL_INSTALL_DIR} -DIPCL_SHARED=OFF
    BUILD_COMMAND CMAKE_BUILD_PARALLEL_LEVEL=${N_CORES} cmake --build .
)

include_directories(${IPCL_INSTALL_DIR}/include/ipcl)
include_directories(${IPCL_INSTALL_DIR}/include/ipcl/ippcrypto)


add_library(compresslwe serialization.cpp defines.cpp library.cpp utils.cpp)
add_executable(tests tests.cpp)

find_package(OpenMP REQUIRED)

add_dependencies(tests ipcl)
add_dependencies(compresslwe ipcl)

target_link_libraries(tests compresslwe  ${IPCL_INSTALL_DIR}/lib64/ipcl/libipcl.a OpenMP::OpenMP_CXX ssl crypto)

install(TARGETS compresslwe EXPORT compresslwe)
install(EXPORT compresslwe DESTINATION ${CMAKE_INSTALL_PREFIX})

install(FILES ${IPCL_INSTALL_DIR}/lib64/ipcl/libipcl.a DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

