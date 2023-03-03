set(LLVM_TARGET_DEFINITIONS Tiling.td)
mlir_tablegen(Tiling.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(ConcretelangFHELinalgTilingPassIncGen)
add_dependencies(mlir-headers ConcretelangFHELinalgTilingPassIncGen)
