add_mlir_dialect_library(
  TracingDialect
  TracingDialect.cpp
  TracingOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/Tracing
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR)

target_link_libraries(TracingDialect PUBLIC MLIRIR)
