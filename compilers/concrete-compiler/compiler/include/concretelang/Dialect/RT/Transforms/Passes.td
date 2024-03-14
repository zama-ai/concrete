#ifndef MLIR_DIALECT_RT_TRANSFORMS_PASSES
#define MLIR_DIALECT_RT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HoistAwaitFuturePass : Pass<"hoist-await-future", "mlir::func::FuncOp"> {
  let summary = "Hoists RT.await_future operations whose results are yielded "
                "by scf.for operations out of the loops";
  let constructor = "mlir::concretelang::createHoistAwaitFuturePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect", "mlir::concretelang::RT::RTDialect"
  ];
}

#endif // MLIR_DIALECT_RT_TRANSFORMS_PASSES
