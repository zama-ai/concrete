#ifndef ZAMALANG_DIALECT_BConcrete_IR_BConcrete_DIALECT
#define ZAMALANG_DIALECT_BConcrete_IR_BConcrete_DIALECT

include "mlir/IR/OpBase.td"

def BConcrete_Dialect : Dialect {
  let name = "BConcrete";
  let summary = "Bufferized concrete dialect";
  let description = [{
    A dialect for representation of bufferized concrete operations on fully homomorphic ciphertext.
  }];
  let cppNamespace = "::mlir::concretelang::BConcrete";
}

#endif
