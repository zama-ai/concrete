add_mlir_dialect_library(HLFHELinalgDialect
  HLFHELinalgDialect.cpp
  HLFHELinalgOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/HLFHELinalg

  DEPENDS
  MLIRHLFHELinalgOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR)

target_link_libraries(HLFHELinalgDialect PUBLIC MLIRIR)
