set(LLVM_TARGET_DEFINITIONS Patterns.td)
mlir_tablegen(Patterns.h.inc -gen-rewriters -name TFHE)
add_public_tablegen_target(TFHEToConcretePatternsIncGen)

add_concretelang_doc(Patterns TFHEToConcretePatterns concretelang/ -gen-pass-doc)
