//===- GLWEDialect.td - GLWE dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWE_DIALECT
#define CONCRETELANG_DIALECT_GLWE_IR_GLWE_DIALECT

include "mlir/IR/OpBase.td"

def GLWE_Dialect : Dialect {
  let name = "glwe";
  let summary = "GLWE Dialect";
  let description = [{
    A dialect to reprensent GLWE types and operators as building blocks for highest level homomorphic construction.
  }];
  let cppNamespace = "::mlir::concretelang::GLWE";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif
