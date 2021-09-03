set(LLVM_TARGET_DEFINITIONS MANP.td)
mlir_tablegen(MANP.h.inc -gen-pass-decls -name Analysis)
mlir_tablegen(MANP.capi.h.inc -gen-pass-capi-header --prefix Analysis)
mlir_tablegen(MANP.capi.cpp.inc -gen-pass-capi-impl --prefix Analysis)
add_public_tablegen_target(MANPPassIncGen)

add_mlir_doc(MANP GeneralPasses ./ -gen-pass-doc)
