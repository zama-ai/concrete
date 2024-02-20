#ifndef CONCRETELANG_DIALECT_OPTIMIZER_TRANSFORMS_PASSES
#define CONCRETELANG_DIALECT_OPTIMIZER_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def OptimizerPartitionFrontierMaterializationPass
    : Pass<"optimizer-partition-frontier-materialization",
           "::mlir::func::FuncOp"> {
  let summary =
      "Inserts Optimizer.partition_frontier operations between FHE operations "
      "that were explicitly marked by the optimizer as belonging to separate "
      "partitions via an extra conversion key in the optimizer solution.";

  let constructor = "mlir::concretelang::"
                    "createOptimizerPartitionFrontierMaterializationPass()";
  let options = [];
  let dependentDialects = [
    "mlir::concretelang::FHE::FHEDialect",
    "mlir::concretelang::Optimizer::OptimizerDialect"
  ];
}

#endif
