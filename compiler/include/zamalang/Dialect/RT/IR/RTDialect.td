#ifndef ZAMALANG_DIALECT_RT_IR_RT_DIALECT
#define ZAMALANG_DIALECT_RT_IR_RT_DIALECT

include "mlir/IR/OpBase.td"

def RT_Dialect : Dialect {
  let name = "RT";
  let summary = "Runtime dialect";
  let description = [{
    A dialect for representation the abstraction needed for the runtime.
  }];
  let cppNamespace = "::mlir::zamalang::RT";
}

#endif
