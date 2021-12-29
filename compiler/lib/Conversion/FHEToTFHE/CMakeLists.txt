add_mlir_dialect_library(FHEToTFHE
  FHEToTFHE.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHE

  DEPENDS
  FHEDialect
  FHEToTFHEPatternsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRMath)

target_link_libraries(FHEToTFHE PUBLIC MLIRIR)
