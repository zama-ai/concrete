//===- OptimizerDialect.td - Optimizer dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_OPTIMIZER_IR_OPTIMIZER_DIALECT
#define CONCRETELANG_DIALECT_OPTIMIZER_IR_OPTIMIZER_DIALECT

include "mlir/IR/OpBase.td"

def Optimizer_Dialect : Dialect {
  let name = "Optimizer";
  let summary = "Auxiliary operations for the interaction with the optimizer";
  let cppNamespace = "::mlir::concretelang::Optimizer";
}

#endif
