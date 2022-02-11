set(LLVM_TARGET_DEFINITIONS Bufferize.td)
mlir_tablegen(Bufferize.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(ConcretelangTransformsPassIncGen)
