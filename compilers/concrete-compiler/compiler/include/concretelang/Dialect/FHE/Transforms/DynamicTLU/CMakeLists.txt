set(LLVM_TARGET_DEFINITIONS DynamicTLU.td)
mlir_tablegen(DynamicTLU.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(DynamicTLUPassIncGen)
add_dependencies(mlir-headers DynamicTLUPassIncGen)
