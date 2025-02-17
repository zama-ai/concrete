set(LLVM_TARGET_DEFINITIONS Transforms.td)
mlir_tablegen(Transforms.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(ConcretelangGLWETransformsPassIncGen)
add_dependencies(mlir-headers ConcretelangGLWETransformsPassIncGen)
