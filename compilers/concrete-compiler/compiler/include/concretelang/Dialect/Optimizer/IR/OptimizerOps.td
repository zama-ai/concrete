//===- OptimizerOps.td - Optimizer dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_OPTIMIZER_IR_OPTIMIZER_OPS
#define CONCRETELANG_DIALECT_OPTIMIZER_IR_OPTIMIZER_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "concretelang/Dialect/Optimizer/IR/OptimizerDialect.td"

class Optimizer_Op<string mnemonic, list<Trait> traits = []> :
    Op<Optimizer_Dialect, mnemonic, traits>;

def Optimizer_PartitionFrontierOp : Optimizer_Op<"partition_frontier", [Pure]> {
    let summary = "Models an explicit edge between two partitions";

    let description = [{
        Models an explicit edge between two partitions in the solution
        determined by the optimizer requiring a key change between the
        encrypted values of the operand and the encrypted values of
        the result.
    }];

    let arguments = (ins
        AnyType:$input,
        I64Attr:$inputKeyID,
        I32Attr:$outputKeyID
    );

    let results = (outs AnyType);
}

#endif
