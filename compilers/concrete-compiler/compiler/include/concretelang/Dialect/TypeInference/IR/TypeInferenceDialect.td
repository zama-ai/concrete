#ifndef CONCRETELANG_DIALECT_TYPEINFERENCE_IR_TYPEINFERENCE_DIALECT
#define CONCRETELANG_DIALECT_TYPEINFERENCE_IR_TYPEINFERENCE_DIALECT

include "mlir/IR/OpBase.td"

def TypeInference_Dialect : Dialect {
  let name = "TypeInference";
  let summary = "TypeInference dialect";
  let description = [{
    A dialect for debugging type inference
  }];
  let cppNamespace = "::mlir::concretelang::TypeInference";
}

#endif
