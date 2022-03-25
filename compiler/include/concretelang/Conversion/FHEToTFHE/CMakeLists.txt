set(LLVM_TARGET_DEFINITIONS Patterns.td)
mlir_tablegen(Patterns.h.inc -gen-rewriters -name FHE)
add_public_tablegen_target(FHEToTFHEPatternsIncGen)
add_dependencies(mlir-headers FHEToTFHEPatternsIncGen)

add_concretelang_doc(Patterns FHEToTFHEPatterns concretelang/ -gen-pass-doc)
