set(LLVM_TARGET_DEFINITIONS Transforms.td)
mlir_tablegen(Transforms.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(ConcretelangTFHETransformsPassIncGen)
add_dependencies(mlir-headers ConcretelangTFHETransformsPassIncGen)
