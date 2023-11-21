add_compile_options(-fexceptions)

add_mlir_library(
  ConcretelangClientLib
  ClientLib.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/ClientLib
  ${PROJECT_SOURCE_DIR}/include/concretelang/Common
  LINK_LIBS
  ConcretelangCommon
  PUBLIC
  concrete_cpu
  concrete-protocol)

target_include_directories(ConcretelangClientLib PUBLIC ${CONCRETE_CPU_INCLUDE_DIR})
