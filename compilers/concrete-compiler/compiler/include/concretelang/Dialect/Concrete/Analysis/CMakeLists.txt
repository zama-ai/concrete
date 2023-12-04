set(LLVM_TARGET_DEFINITIONS Analysis.td)
mlir_tablegen(Analysis.h.inc -gen-pass-decls -name Analysis)
add_public_tablegen_target(ConcretelangConcreteAnalysisPassIncGen)
add_dependencies(mlir-headers ConcretelangConcreteAnalysisPassIncGen)
