add_mlir_dialect_library(
  TFHEToConcrete
  TFHEToConcrete.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/TFHE
  DEPENDS
  TFHEDialect
  ConcreteDialect
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRMathDialect)

target_link_libraries(TFHEToConcrete PUBLIC MLIRIR)
