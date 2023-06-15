//===- ConcreteDialect.td - Concrete dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_Concrete_IR_Concrete_DIALECT
#define CONCRETELANG_DIALECT_Concrete_IR_Concrete_DIALECT

include "mlir/IR/OpBase.td"

def Concrete_Dialect : Dialect {
  let name = "Concrete";
  let summary = "Low Level Fully Homomorphic Encryption dialect";
  let description = [{
    A dialect for representation of low level operation on fully homomorphic ciphertext.
  }];
  let cppNamespace = "::mlir::concretelang::Concrete";
  let useDefaultTypePrinterParser = 1;
}

#endif
