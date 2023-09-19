#ifndef CONCRETELANG_FHELINALG_TILING_PASS
#define CONCRETELANG_FHELINALG_TILING_PASS

include "mlir/Pass/PassBase.td"

def CollapseParallelLoops : Pass<"collapse-parallel-loops", "mlir::ModuleOp"> {
  let summary =
      "Coalesce nested scf.for operations that are marked with "
      "the custom attribute parallel = true into a single scf.for "
      "loop which can subsequently be converted to scf.parallel.";
  let constructor = "mlir::concretelang::createCollapseParallelLoops()";
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

def ForLoopToParallel : Pass<"for-loop-to-parallel", "mlir::ModuleOp"> {
  let summary =
      "Transform scf.for marked with the custom attribute parallel = true loop "
      "to scf.parallel after the bufferization";
  let constructor = "mlir::concretelang::createForLoopToParallel()";
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

def Batching : Pass<"concrete", "mlir::ModuleOp"> {
  let summary =
      "Hoists operation for which a batched version exists out of loops applying "
      "the operation to values stored in a tensor.";
  let constructor = "mlir::concretelang::createBatchingPass()";
}

def SCFForallToSCFFor : Pass<"scf-forall-to-scf-for", "mlir::ModuleOp"> {
  let summary = "Replaces all occurrences of scf.forall with equivalent loop nests of scf.for";
  let description = [{ Replaces all occurrences of scf.forall with equivalent loop nests of scf.for }];
  let constructor = "mlir::concretelang::createSCFForallToSCFForPass()";
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

def TensorEmptyToBufferizationAlloc : Pass<"tensor-empty-to-bufferization-alloc", "mlir::ModuleOp"> {
  let summary = "Replaces all occurrences of tensor.empty with bufferization.alloc_tensor";
  let description = [{ Replaces all occurrences of tensor.empty with bufferization.alloc_tensor }];
  let constructor = "mlir::concretelang::createTensorEmptyToBufferizationAllocPass()";
  let dependentDialects = ["mlir::tensor::TensorDialect", "mlir::bufferization::BufferizationDialect"];
}

#endif
