add_compile_options(-fexceptions -fsized-deallocation)

add_mlir_library(
  ConcretelangSupport
  Pipeline.cpp
  CompilationFeedback.cpp
  CompilerEngine.cpp
  TFHECircuitKeys.cpp
  Encodings.cpp
  V0Parameters.cpp
  ProgramInfoGeneration.cpp
  logging.cpp
  LLVMEmitFile.cpp
  Utils.cpp
  DEPENDS
  mlir-headers
  concrete-protocol
  LINK_LIBS
  PUBLIC
  FHELinalgDialect
  GLWEDialect
  FHELinalgDialectTransforms
  FHETensorOpsToLinalg
  FHEDialectTransforms
  FHEToTFHECrt
  FHEToTFHEScalar
  SimulateTFHE
  ExtractSDFGOps
  MLIRLowerableDialectsToLLVM
  FHEDialectAnalysis
  FHEDialectTransforms
  RTDialectAnalysis
  ConcretelangTransforms
  ConcretelangConcreteTransforms
  ConcretelangSDFGTransforms
  ConcretelangSDFGInterfaces
  LinalgExtras
  TFHEDialectTransforms
  concrete_optimizer
  MLIRExecutionEngine
  ${LLVM_PTHREAD_LIB}
  ConcretelangRuntime
  ConcretelangClientLib
  ConcretelangServerLib
  TFHEDialectAnalysis
  ConcreteDialectAnalysis)

target_include_directories(ConcretelangSupport PUBLIC ${CONCRETE_CPU_INCLUDE_DIR})
