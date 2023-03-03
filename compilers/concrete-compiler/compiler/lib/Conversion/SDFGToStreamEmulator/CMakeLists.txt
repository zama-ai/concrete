add_mlir_dialect_library(
  SDFGToStreamEmulator
  SDFGToStreamEmulator.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/SDFG
  DEPENDS
  SDFGDialect
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransforms)

target_link_libraries(SDFGToStreamEmulator PUBLIC SDFGDialect MLIRIR)
