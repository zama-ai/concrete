#ifndef CONCRETELANG_FHE_BOOLEAN_PASS
#define CONCRETELANG_FHE_BOOLEAN_PASS

include "mlir/Pass/PassBase.td"

def FHEBooleanTransform : Pass<"fhe-boolean-transform"> {
  let summary = "Transform FHE boolean operations to integer operations";
  let constructor = "mlir::concretelang::createFHEBooleanTransformPass()";
  let options = [];
  let dependentDialects = [ "mlir::concretelang::FHE::FHEDialect" ];
}

#endif
