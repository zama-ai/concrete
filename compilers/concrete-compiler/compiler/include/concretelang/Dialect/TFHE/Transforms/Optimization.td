#ifndef CONCRETELANG_TFHE_OPTIMIZATION_PASS
#define CONCRETELANG_TFHE_OPTIMIZATION_PASS

include "mlir/Pass/PassBase.td"

def TFHEOptimization : Pass<"tfhe-optimization"> {
  let summary = "Optimize TFHE operations";
  let constructor = "mlir::concretelang::createTFHEOptimizationPass()";
  let options = [];
  let dependentDialects = [ "mlir::concretelang::TFHE::TFHEDialect" ];
}

#endif
