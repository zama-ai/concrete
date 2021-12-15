add_mlir_library(HLFHELinalgDialectTransforms
  Tiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/zamalang/Dialect/HLFHELinalg

  DEPENDS
  HLFHELinalgDialect
  ZamalangHLFHELinalgTilingPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  HLFHELinalgDialect)

target_link_libraries(HLFHELinalgDialectTransforms PUBLIC MLIRIR)
