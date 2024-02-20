add_mlir_dialect_library(
  FHEToTFHEScalar
  FHEToTFHEScalar.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHE
  DEPENDS
  FHEDialect
  OptimizerDialect
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRMathDialect)

target_link_libraries(FHEToTFHEScalar PUBLIC MLIRIR)
