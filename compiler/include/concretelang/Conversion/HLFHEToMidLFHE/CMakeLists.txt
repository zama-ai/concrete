set(LLVM_TARGET_DEFINITIONS Patterns.td)
mlir_tablegen(Patterns.h.inc -gen-rewriters -name HLFHE)
add_public_tablegen_target(HLFHEToMidLFHEPatternsIncGen)

add_concretelang_doc(Patterns HLFHEToMidLFHEPatterns concretelang/ -gen-pass-doc)
