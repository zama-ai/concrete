add_mlir_dialect_library(
  SimulateTFHE
  SimulateTFHE.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/TFHE
  DEPENDS
  TFHEDialect
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRMathDialect)

target_link_libraries(SimulateTFHE PUBLIC MLIRIR)
