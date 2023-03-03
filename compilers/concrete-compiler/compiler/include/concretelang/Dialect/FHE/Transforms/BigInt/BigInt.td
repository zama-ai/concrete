#ifndef CONCRETELANG_FHE_BIGINT_PASS
#define CONCRETELANG_FHE_BIGINT_PASS

include "mlir/Pass/PassBase.td"

def FHEBigIntTransform : Pass<"fhe-big-int-transform"> {
  let summary = "Transform FHE operations on big integer into operations on chunks of small integer";
  let constructor = "mlir::concretelang::createFHEBigIntTransformPass()";
  let options = [];
  let dependentDialects = [ "mlir::concretelang::FHE::FHEDialect" ];
}

#endif
