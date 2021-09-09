add_mlir_dialect_library(LowLFHEUnparametrize
  LowLFHEUnparametrize.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/zamalang/Dialect/HLFHE

  DEPENDS
  LowLFHEDialect
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
)

target_link_libraries(LowLFHEUnparametrize PUBLIC MLIRIR)
