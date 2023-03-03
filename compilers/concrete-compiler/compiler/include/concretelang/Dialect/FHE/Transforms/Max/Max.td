#ifndef CONCRETELANG_FHE_MAX_PASS
#define CONCRETELANG_FHE_MAX_PASS

include "mlir/Pass/PassBase.td"

def FHEMaxTransform : Pass<"fhe-max-transform"> {
  let summary = "Transform max operation to basic operations";
  let constructor = "mlir::concretelang::createFHEMaxTransformPass()";
  let options = [];
  let dependentDialects = [ "mlir::concretelang::FHE::FHEDialect" ];
}

#endif
