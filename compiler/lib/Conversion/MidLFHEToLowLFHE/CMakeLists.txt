add_mlir_dialect_library(MidLFHEToLowLFHE
  MidLFHEToLowLFHE.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/MidLFHE

  DEPENDS
  MidLFHEDialect
  LowLFHEDialect
  MidLFHEToLowLFHEPatternsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRMath)

target_link_libraries(MidLFHEToLowLFHE PUBLIC MLIRIR)
