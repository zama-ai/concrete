add_mlir_dialect_library(
  ExtractSDFGOps
  ExtractSDFGOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHE
  DEPENDS
  SDFGDialect
  ConcretelangSDFGInterfaces
  mlir-headers
  LINK_LIBS
  SDFGDialect
  ConcretelangSDFGInterfaces
  PUBLIC
  MLIRIR
  MLIRSCFUtils
  MLIRTransforms)

target_link_libraries(ExtractSDFGOps PUBLIC MLIRIR)
