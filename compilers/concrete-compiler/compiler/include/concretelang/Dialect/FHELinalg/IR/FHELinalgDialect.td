#ifndef CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalg_DIALECT
#define CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalg_DIALECT

include "mlir/IR/OpBase.td"

def FHELinalg_Dialect : Dialect {
  let name = "FHELinalg";
  let summary = "High Level Fully Homorphic Encryption Linalg dialect";
  let description = [{
    A dialect for representation of high level linalg operations on fully homomorphic ciphertexts.
  }];
  let cppNamespace = "::mlir::concretelang::FHELinalg";
  let useFoldAPI = kEmitRawAttributesFolder;
}

#endif
