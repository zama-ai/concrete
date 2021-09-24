add_mlir_library(ZamalangSupport
  Pipeline.cpp
  Jit.cpp
  CompilerEngine.cpp
  V0Parameters.cpp
  V0Curves.cpp
  ClientParameters.cpp
  KeySet.cpp
  logging.cpp
  Jit.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/zamalang/Support

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  HLFHETensorOpsToLinalg
  HLFHEToMidLFHE
  LowLFHEUnparametrize
  MLIRLowerableDialectsToLLVM
  HLFHEDialectAnalysis

  MLIRExecutionEngine
  ${LLVM_PTHREAD_LIB}
  
  ZamalangRuntime
)
