add_mlir_dialect_library(
  TFHEDialect
  TFHEDialect.cpp
  TFHEOps.cpp
  TFHETypes.cpp
  TFHEParameters.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/TFHE
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR)

target_link_libraries(TFHEDialect PUBLIC MLIRIR)
