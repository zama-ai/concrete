add_mlir_library(ConcretelangTransforms
  Bufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Transforms

  DEPENDS
  MLIRTransforms
  mlir-headers
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRef
  MLIRTransforms
)
