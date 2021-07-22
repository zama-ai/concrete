add_mlir_dialect_library(HLFHETensorOpsToLinalg
  TensorOpsToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/zamalang/Dialect/HLFHE

  DEPENDS
  HLFHEDialect
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  HLFHEDialect)

target_link_libraries(HLFHEDialect PUBLIC MLIRIR)
