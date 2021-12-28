//===- MidLFHEDialect.td - MidLFHE dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_MidLFHE_IR_MidLFHE_DIALECT
#define CONCRETELANG_DIALECT_MidLFHE_IR_MidLFHE_DIALECT

include "mlir/IR/OpBase.td"

def MidLFHE_Dialect : Dialect {
  let name = "MidLFHE";
  let summary = "High Level Fully Homorphic Encryption dialect";
  let description = [{
    A dialect for representation of high level operation on fully homomorphic ciphertext.
  }];
  let cppNamespace = "::mlir::concretelang::MidLFHE";
}

#endif
