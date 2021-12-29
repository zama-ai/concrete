add_mlir_library(ConcretelangSupport
  Error.cpp
  Pipeline.cpp
  Jit.cpp
  CompilerEngine.cpp
  JitCompilerEngine.cpp
  LambdaArgument.cpp
  V0Parameters.cpp
  V0Curves.cpp
  ClientParameters.cpp
  KeySet.cpp
  logging.cpp
  Jit.cpp
  KeySetCache.cpp
  LLVMEmitFile.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Support

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  FHELinalgDialect
  FHELinalgDialectTransforms
  FHETensorOpsToLinalg
  FHEToTFHE
  ConcreteUnparametrize
  MLIRLowerableDialectsToLLVM
  FHEDialectAnalysis
  RTDialectAnalysis

  MLIRExecutionEngine
  ${LLVM_PTHREAD_LIB}
  
  ConcretelangRuntime
)
