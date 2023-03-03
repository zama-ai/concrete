add_mlir_dialect_library(
  FHELinalgDialect
  FHELinalgDialect.cpp
  FHELinalgOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHELinalg
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR)

target_link_libraries(FHELinalgDialect PUBLIC MLIRIR)
