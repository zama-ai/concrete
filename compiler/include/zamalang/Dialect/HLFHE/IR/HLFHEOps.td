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

// Generates an encrypted zero constant
def ZeroOp : HLFHE_Op<"zero"> {
    let arguments = (ins);
    let results = (outs EncryptedIntegerType:$out);
}

def AddEintIntOp : HLFHE_Op<"add_eint_int"> {
    let arguments = (ins EncryptedIntegerType:$a, AnyInteger:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::zamalang::HLFHE::verifyAddEintIntOp(*this);
    }];
}

def AddEintOp : HLFHE_Op<"add_eint"> {
    let arguments = (ins EncryptedIntegerType:$a, EncryptedIntegerType:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::zamalang::HLFHE::verifyAddEintOp(*this);
    }];
}

def SubIntEintOp : HLFHE_Op<"sub_int_eint"> {
    let arguments = (ins AnyInteger:$a, EncryptedIntegerType:$b);
    let results = (outs EncryptedIntegerType);

    let verifier = [{
        return ::mlir::zamalang::HLFHE::verifySubIntEintOp(*this);
    }];
}

def MulEintIntOp : HLFHE_Op<"mul_eint_int"> {
    let arguments = (ins EncryptedIntegerType:$a, AnyInteger:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::zamalang::HLFHE::verifyMulEintIntOp(*this);
    }];
}

def ApplyLookupTableEintOp : HLFHE_Op<"apply_lookup_table"> {
    let arguments = (ins EncryptedIntegerType:$ct,
        MemRefOf<[AnyInteger]>:$l_cst);
    let results = (outs EncryptedIntegerType);

    let verifier = [{
        return ::mlir::zamalang::HLFHE::verifyApplyLookupTable(*this);
    }];
}

// Tensor operations

// Dot product
def Dot : HLFHE_Op<"dot_eint_int"> {
    let arguments = (ins
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred, HasAnyRankOfPred<[1]>]>>:$lhs,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred, HasAnyRankOfPred<[1]>]>>:$rhs);
    let results = (outs EncryptedIntegerType:$out);
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
