set(LLVM_TARGET_DEFINITIONS Boolean.td)
mlir_tablegen(Boolean.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(ConcretelangFHEBooleanPassIncGen)
add_dependencies(mlir-headers ConcretelangFHEBooleanPassIncGen)
