add_mlir_dialect_library(RTDialectTransforms
  BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/RT

  DEPENDS
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

