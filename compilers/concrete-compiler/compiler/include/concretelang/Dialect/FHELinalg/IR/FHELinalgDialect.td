//===- FHELinalgDialect.td - FHELinalg dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalg_DIALECT
#define CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalg_DIALECT

include "mlir/IR/OpBase.td"

def FHELinalg_Dialect : Dialect {
  let name = "FHELinalg";
  let summary = "High Level Fully Homomorphic Encryption Linalg dialect";
  let description = [{
    A dialect for representation of high level linalg operations on fully homomorphic ciphertexts.
  }];
  let cppNamespace = "::mlir::concretelang::FHELinalg";
}

#endif
