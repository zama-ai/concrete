add_mlir_dialect_library(
  TracingToCAPI
  TracingToCAPI.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/Tracing
  DEPENDS
  TracingDialect
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransforms)

target_link_libraries(TracingToCAPI PUBLIC TracingDialect MLIRIR)
