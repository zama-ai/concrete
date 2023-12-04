add_mlir_dialect_library(
  FHEDialect
  FHEDialect.cpp
  FHEOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHE
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
  FHEInterfaces)

target_link_libraries(FHEDialect PUBLIC MLIRIR)
