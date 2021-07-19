//===- HLFHEOps.td - High level FHE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZAMALANG_DIALECT_HLFHE_IR_HLFHE_OPS
#define ZAMALANG_DIALECT_HLFHE_IR_HLFHE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.td"
include "zamalang/Dialect/HLFHE/IR/HLFHETypes.td"

class HLFHE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<HLFHE_Dialect, mnemonic, traits>;

def AddEintIntOp : HLFHE_Op<"add_eint_int"> {
    let arguments = (ins EncryptedIntegerType:$a, AnyInteger:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];
}

def AddEintOp : HLFHE_Op<"add_eint"> {
    let arguments = (ins EncryptedIntegerType:$a, EncryptedIntegerType:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];
}

def MulEintIntOp : HLFHE_Op<"mul_eint_int"> {
    let arguments = (ins EncryptedIntegerType:$a, AnyInteger:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];
}

// Tensor operations

// Dot product
def Dot : HLFHE_Op<"dot_eint_int", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    // Output memref is passed as the last argument; Input and output
    // relationships are modeled through MemoryEffectsOpInterface`
    let arguments = (ins
        Type<And<[StaticShapeMemRefOf<[EncryptedIntegerType]>.predicate, HasAnyRankOfPred<[1]>]>>:$lhs,
        Type<And<[StaticShapeMemRefOf<[AnyInteger]>.predicate, HasAnyRankOfPred<[1]>]>>:$rhs,
        Type<And<[StaticShapeMemRefOf<[EncryptedIntegerType]>.predicate, HasAnyRankOfPred<[0]>]>>:$out);
    let verifier = [{
        if(::mlir::failed(
              mlir::verifyCompatibleShape(
                  lhs().getType(),
                  rhs().getType()))) {
            return this->emitOpError("arguments have incompatible shapes");
        }

        return ::mlir::success();
    }];
}
#endif
