//===- SDFGDialect.td - SDFG dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_SDFG_IR_SDFG_DIALECT
#define CONCRETELANG_DIALECT_SDFG_IR_SDFG_DIALECT

include "mlir/IR/OpBase.td"

def SDFG_Dialect : Dialect {
  let name = "SDFG";
  let summary = "Dialect for the construction of static data flow graphs";
  let description = [{
    A dialect for the construction of static data flow graphs. The
    data flow graph is composed of a set of processes, connected
    through data streams. Special streams allow for data to be
    injected into and to be retrieved from the data flow graph.
  }];
  let cppNamespace = "::mlir::concretelang::SDFG";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif
