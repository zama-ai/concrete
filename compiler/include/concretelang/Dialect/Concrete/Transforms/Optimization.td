#ifndef CONCRETELANG_CONCRETE_OPTIMIZATION_PASS
#define CONCRETELANG_CONCRETE_OPTIMIZATION_PASS

include "mlir/Pass/PassBase.td"

def ConcreteOptimization : Pass<"concrete-optimization"> {
  let summary = "Optimize Concrete operations";
  let constructor = "mlir::concretelang::createConcreteOptimizationPass()";
  let options = [];
  let dependentDialects = [ "mlir::concretelang::Concrete::ConcreteDialect" ];
}

#endif
