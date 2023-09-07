#ifndef CONCRETELANG_FHE_ENCRYPTED_MUL_TO_DOUBLE_TLU_PASS
#define CONCRETELANG_FHE_ENCRYPTED_MUL_TO_DOUBLE_TLU_PASS

include "mlir/Pass/PassBase.td"

def EncryptedMulToDoubleTLU : Pass<"EncryptedMulToDoubleTLU", "::mlir::func::FuncOp"> {
  let summary = "Replaces encrypted multiplication with a double table lookup.";
  let constructor = "mlir::concretelang::createEncryptedMulToDoubleTLUPass()";
}

#endif
