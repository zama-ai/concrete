#ifndef CONCRETELANG_TFHE_TRANSFORMS_PASS
#define CONCRETELANG_TFHE_TRANSFORMS_PASS

include "mlir/Pass/PassBase.td"

def GLWEInjectDefaultVariances : Pass<"glwe-inject-default-variances", "mlir::ModuleOp"> {
  let summary = "Inject default variance in GLWE operations";
  let constructor = "mlir::concretelang::GLWE::createInjectDefaultVariancesPass()";
  let options = [];
  let dependentDialects = [ "mlir::concretelang::GLWE::GLWEDialect" ];
}

def GLWEPropagateVariances : Pass<"glwe-propagate-variances", "mlir::ModuleOp"> {
  let summary = "Propagate variances through GLWE operations";
  let constructor = "mlir::concretelang::GLWE::createPropagatetVariancesPass()";
  let options = [];
  let dependentDialects = [ "mlir::concretelang::GLWE::GLWEDialect" ];
}

#endif
