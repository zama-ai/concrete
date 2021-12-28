add_mlir_library(HLFHELinalgDialectTransforms
  Tiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/HLFHELinalg

  DEPENDS
  HLFHELinalgDialect
  ConcretelangHLFHELinalgTilingPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  HLFHELinalgDialect)

target_link_libraries(HLFHELinalgDialectTransforms PUBLIC MLIRIR)
