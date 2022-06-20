//===-- Passes.td - pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES
#define MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AddRuntimeContext : Pass<"add-runtime-context", "mlir::ModuleOp"> {
  let summary = "Add the runtime context argument";
  let constructor = "mlir::concretelang::createAddRuntimeContext()";
}

def EliminateCRTOps : Pass<"eliminate-bconcrete-crt-ops", "mlir::func::FuncOp"> {
  let summary = "Eliminate the crt bconcrete operators.";
  let constructor = "mlir::concretelang::createEliminateCRTOpsPass()";
}

#endif // MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES
