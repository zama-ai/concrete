add_mlir_library(ZamalangSupport
  CompilerTools.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/zamalang/Support

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  HLFHETensorOpsToLinalg
  HLFHEToMidLFHE
  MLIRLowerableDialectsToLLVM

  MLIRExecutionEngine
  ${LLVM_PTHREAD_LIB})