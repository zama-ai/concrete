add_mlir_dialect_library(
  OptimizerDialect
  OptimizerDialect.cpp
  OptimizerOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/Optimizer
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR)

target_link_libraries(OptimizerDialect PUBLIC MLIRIR)
