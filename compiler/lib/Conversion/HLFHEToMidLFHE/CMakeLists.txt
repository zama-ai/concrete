add_mlir_dialect_library(HLFHEToMidLFHE
  HLFHEToMidLFHE.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/HLFHE

  DEPENDS
  HLFHEDialect
  HLFHEToMidLFHEPatternsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRMath)

target_link_libraries(HLFHEToMidLFHE PUBLIC MLIRIR)
