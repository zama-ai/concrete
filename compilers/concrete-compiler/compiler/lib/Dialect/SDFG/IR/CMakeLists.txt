add_mlir_dialect_library(
  SDFGDialect
  SDFGDialect.cpp
  SDFGOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/SDFG
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR)

target_link_libraries(SDFGDialect PUBLIC MLIRIR)
