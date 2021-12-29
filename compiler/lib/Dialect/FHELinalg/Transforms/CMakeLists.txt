add_mlir_library(FHELinalgDialectTransforms
  Tiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHELinalg

  DEPENDS
  FHELinalgDialect
  ConcretelangFHELinalgTilingPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  FHELinalgDialect)

target_link_libraries(FHELinalgDialectTransforms PUBLIC MLIRIR)
