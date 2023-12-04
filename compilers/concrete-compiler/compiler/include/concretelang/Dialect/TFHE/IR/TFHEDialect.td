//===- TFHEDialect.td - TFHE dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_TFHE_IR_TFHE_DIALECT
#define CONCRETELANG_DIALECT_TFHE_IR_TFHE_DIALECT

include "mlir/IR/OpBase.td"

def TFHE_Dialect : Dialect {
  let name = "TFHE";
  let summary = "High Level Fully Homomorphic Encryption dialect";
  let description = [{
    A dialect for representation of high level operation on fully homomorphic ciphertext.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::concretelang::TFHE";
}

#endif
