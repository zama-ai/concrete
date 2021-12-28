set(LLVM_TARGET_DEFINITIONS Patterns.td)
mlir_tablegen(Patterns.h.inc -gen-rewriters -name MidLFHE)
add_public_tablegen_target(MidLFHEToLowLFHEPatternsIncGen)

add_concretelang_doc(Patterns MidLFHEToLowLFHEPatterns concretelang/ -gen-pass-doc)
