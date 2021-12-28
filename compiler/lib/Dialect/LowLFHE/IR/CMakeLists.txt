add_mlir_dialect_library(LowLFHEDialect
  LowLFHEDialect.cpp
  LowLFHEOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/LowLFHE

  DEPENDS
  MLIRLowLFHEOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR)

target_link_libraries(LowLFHEDialect PUBLIC MLIRIR)
