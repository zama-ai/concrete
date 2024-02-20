add_mlir_dialect_library(
  FHETensorOpsToLinalg
  TensorOpsToLinalg.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHE
  DEPENDS
  FHEDialect
  FHELinalgDialect
  OptimizerDialect
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
  FHEDialect
  FHELinalgDialect
  OptimizerDialect)

target_link_libraries(FHEDialect PUBLIC MLIRIR)
