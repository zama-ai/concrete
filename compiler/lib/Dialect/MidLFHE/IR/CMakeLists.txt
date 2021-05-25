add_mlir_dialect_library(MidLFHEDialect
  MidLFHEDialect.cpp
  MidLFHEOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/zamalang/Dialect/MidLFHE

  DEPENDS
  MLIRMidLFHEOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR)

target_link_libraries(MidLFHEDialect PUBLIC MLIRIR)
