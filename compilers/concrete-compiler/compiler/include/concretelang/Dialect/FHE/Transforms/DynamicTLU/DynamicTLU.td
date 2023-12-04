#ifndef CONCRETELANG_FHE_DYNAMIC_TLU_PASS
#define CONCRETELANG_FHE_DYNAMIC_TLU_PASS

include "mlir/Pass/PassBase.td"

def DynamicTLU : Pass<"DynamicTLU", "::mlir::func::FuncOp"> {
  let summary = "Enable table lookups with luts of arbitrary integer precision.";
  let constructor = "mlir::concretelang::createDynamicTLUPass()";
}

#endif
