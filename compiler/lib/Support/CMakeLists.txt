# not working in ADDITIONAL_HEADER_DIRS
include_directories(${CONCRETE_OPTIMIZER_DIR}/concrete-optimizer-cpp/src/cpp)

add_mlir_library(ConcretelangSupport
  Pipeline.cpp
  Jit.cpp
  CompilerEngine.cpp
  JITSupport.cpp
  LambdaArgument.cpp
  V0Parameters.cpp
  V0Curves.cpp
  V0ClientParameters.cpp
  logging.cpp
  Jit.cpp
  LLVMEmitFile.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Support

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  FHELinalgDialect
  FHELinalgDialectTransforms
  FHETensorOpsToLinalg
  FHEToTFHE
  MLIRLowerableDialectsToLLVM
  FHEDialectAnalysis
  RTDialectAnalysis
  ConcretelangTransforms
  concrete_optimizer

  MLIRExecutionEngine
  ${LLVM_PTHREAD_LIB}

  ConcretelangRuntime
  ConcretelangClientLib
  ConcretelangServerLib
  Concrete
)
