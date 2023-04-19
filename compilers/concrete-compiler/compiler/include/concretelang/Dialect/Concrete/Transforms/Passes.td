#ifndef MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES
#define MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AddRuntimeContext : Pass<"add-runtime-context", "mlir::ModuleOp"> {
  let summary = "Add the runtime context argument";
  let constructor = "mlir::concretelang::createAddRuntimeContext()";
}

#endif // MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES
