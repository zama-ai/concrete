//===- TracingDialect.td - Tracing dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_TRACING_IR_TRACING_DIALECT
#define CONCRETELANG_DIALECT_TRACING_IR_TRACING_DIALECT

include "mlir/IR/OpBase.td"

def Tracing_Dialect : Dialect {
  let name = "Tracing";
  let summary = "Tracing dialect";
  let description = [{
    A dialect to print program values at runtime.
  }];
  let cppNamespace = "::mlir::concretelang::Tracing";
}

#endif
