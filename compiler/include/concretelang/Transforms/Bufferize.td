#ifndef CONCRETELANG_FHELINALG_TILING_PASS
#define CONCRETELANG_FHELINALG_TILING_PASS

include "mlir/Pass/PassBase.td"

def FinalizingBufferize
    : Pass<"concretelang-bufferize", "::mlir::func::FuncOp"> {
  let summary =
      "Marks FHELinalg operations for tiling using a vector of tile sizes";
  let constructor = "mlir::concretelang::createBufferizePass()";
  let options = [];
  let dependentDialects =
      ["mlir::memref::MemRefDialect", "mlir::func::FuncDialect"];
}

def ForLoopToParallel : Pass<"for-loop-to-parallel", "mlir::ModuleOp"> {
  let summary =
      "Transform scf.for marked with the custom attribute parallel = true loop "
      "to scf.parallel after the bufferization";
  let constructor = "mlir::concretelang::createForLoopToParallel()";
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

#endif
