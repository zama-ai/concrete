#ifndef ZAMALANG_DIALECT_LowLFHE_IR_LowLFHE_DIALECT
#define ZAMALANG_DIALECT_LowLFHE_IR_LowLFHE_DIALECT

include "mlir/IR/OpBase.td"

def LowLFHE_Dialect : Dialect {
  let name = "LowLFHE";
  let summary = "Low Level Fully Homorphic Encryption dialect";
  let description = [{
    A dialect for representation of low level operation on fully homomorphic ciphertext.
  }];
  let cppNamespace = "::mlir::zamalang::LowLFHE";
}

#endif
