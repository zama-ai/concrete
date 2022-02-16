add_mlir_dialect_library(FHETensorOpsToLinalg
  TensorOpsToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHE

  DEPENDS
  FHEDialect
  FHELinalgDialect
  ConcretelangConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  FHEDialect
  FHELinalgDialect)

target_link_libraries(FHEDialect PUBLIC MLIRIR)
