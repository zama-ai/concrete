#ifndef CONCRETELANG_DIALECT_TRACING_IR_TRACING_DIALECT
#define CONCRETELANG_DIALECT_TRACING_IR_TRACING_DIALECT

include "mlir/IR/OpBase.td"

def Tracing_Dialect : Dialect {
  let name = "Tracing";
  let summary = "Tracing dialect";
  let description = [{
    A dialect to print program values at runtime.
  }];
  let cppNamespace = "::mlir::concretelang::Tracing";
}

#endif
