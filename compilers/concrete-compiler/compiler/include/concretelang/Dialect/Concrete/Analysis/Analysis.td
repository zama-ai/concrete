#ifndef CONCRETELANG_DIALECT_CONCRETE_ANALYSIS
#define CONCRETELANG_DIALECT_CONCRETE_ANALYSIS

include "mlir/Pass/PassBase.td"

def MemoryUsage : Pass<"MemoryUsage", "::mlir::ModuleOp"> {
  let summary = "Compute memory usage";
  let description = [{
    Computes memory usage per location, and provides those numbers throught the CompilationFeedback.
  }];
}

#endif
