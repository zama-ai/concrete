#ifndef ZAMALANG_DIALECT_HLFHELinalg_IR_HLFHELinalg_DIALECT
#define ZAMALANG_DIALECT_HLFHELinalg_IR_HLFHELinalg_DIALECT

include "mlir/IR/OpBase.td"

def HLFHELinalg_Dialect : Dialect {
  let name = "HLFHELinalg";
  let summary = "High Level Fully Homorphic Encryption Linalg dialect";
  let description = [{
    A dialect for representation of high level linalg operations on fully homomorphic ciphertexts.
  }];
  let cppNamespace = "::mlir::zamalang::HLFHELinalg";
}

#endif
