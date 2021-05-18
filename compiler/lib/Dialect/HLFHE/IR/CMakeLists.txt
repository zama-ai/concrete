add_mlir_dialect_library(HLFHEDialect
  HLFHEDialect.cpp
  HLFHEOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/zamalang/Dialect/HLFHE

  DEPENDS
  MLIRHLFHEOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR)

target_link_libraries(HLFHEDialect PUBLIC MLIRIR)
