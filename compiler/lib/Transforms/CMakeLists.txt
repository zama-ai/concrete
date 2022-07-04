add_mlir_library(ConcretelangTransforms
  ForLoopToParallel.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Transforms

  DEPENDS
  MLIRTransforms
  ConcretelangTransformsBufferizePassIncGen
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRTransforms
)
