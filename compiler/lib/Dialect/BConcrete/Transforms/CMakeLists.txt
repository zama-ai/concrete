add_mlir_dialect_library(ConcretelangBConcreteTransforms
  BufferizableOpInterfaceImpl.cpp
  AddRuntimeContext.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/BConcrete

  DEPENDS
  BConcreteTransformsIncGen
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRBufferization
  MLIRBufferizationTransforms
  MLIRIR
  MLIRMemRef
  MLIRPass
  MLIRTransforms
  )
