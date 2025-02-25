add_compile_options(-fexceptions)

add_mlir_library(
  ConcretelangClientLib
  ClientLib.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/ClientLib
  ${PROJECT_SOURCE_DIR}/include/concretelang/Common
  ${CONCRETE_CPU_INCLUDE_DIR}
  LINK_LIBS
  ConcretelangCommon
  PUBLIC
  rust_deps_bundle
  concrete-protocol)
