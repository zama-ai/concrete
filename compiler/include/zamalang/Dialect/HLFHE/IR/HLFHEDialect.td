//===- HLFHEDialect.td - HLFHE dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZAMALANG_DIALECT_HLFHE_IR_HLFHE_DIALECT
#define ZAMALANG_DIALECT_HLFHE_IR_HLFHE_DIALECT

include "mlir/IR/OpBase.td"

def HLFHE_Dialect : Dialect {
  let name = "HLFHE";
  let summary = "High Level Fully Homorphic Encryption dialect";
  let description = [{
    A dialect for representation of high level operation on fully homomorphic ciphertext.
  }];
  let cppNamespace = "::mlir::zamalang::HLFHE";
}

#endif
