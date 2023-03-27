add_mlir_dialect_library(
  TFHEKeyNormalization
  TFHEKeyNormalization.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/TFHE
  DEPENDS
  TFHEDialect
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransforms)

target_link_libraries(TFHEKeyNormalization PUBLIC MLIRIR)
