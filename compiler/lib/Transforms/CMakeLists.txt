add_mlir_library(ConcretelangTransforms
  Bufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Transforms

  DEPENDS
  MLIRTransforms
  ConcretelangTransformsPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRMemRef
  MLIRTransforms
)

target_link_libraries(FHELinalgDialectTransforms PUBLIC MLIRIR)
