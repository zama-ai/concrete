set(LLVM_TARGET_DEFINITIONS Max.td)
mlir_tablegen(Max.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(ConcretelangFHEMaxPassIncGen)
add_dependencies(mlir-headers ConcretelangFHEMaxPassIncGen)
