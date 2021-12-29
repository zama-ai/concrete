//===- FHEOps.td - High level FHE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_FHE_IR_FHE_OPS
#define CONCRETELANG_DIALECT_FHE_IR_FHE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "concretelang/Dialect/FHE/IR/FHEDialect.td"
include "concretelang/Dialect/FHE/IR/FHETypes.td"

class FHE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<FHE_Dialect, mnemonic, traits>;

// Generates an encrypted zero constant
def ZeroEintOp : FHE_Op<"zero", [NoSideEffect]> {
    let summary = "Return an encryption of 0";

    let arguments = (ins);
    let results = (outs EncryptedIntegerType:$out);
}

def AddEintIntOp : FHE_Op<"add_eint_int"> {

    let summary = "Adds an encrypted integer and a clear integer";

    let arguments = (ins EncryptedIntegerType:$a, AnyInteger:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::concretelang::FHE::verifyAddEintIntOp(*this);
    }];
}

def AddEintOp : FHE_Op<"add_eint"> {

    let summary = "Adds two encrypted integers";

    let arguments = (ins EncryptedIntegerType:$a, EncryptedIntegerType:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::concretelang::FHE::verifyAddEintOp(*this);
    }];
}

def SubIntEintOp : FHE_Op<"sub_int_eint"> {

    let summary = "Substract a clear integer and an encrypted integer";

    let arguments = (ins AnyInteger:$a, EncryptedIntegerType:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, b.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::concretelang::FHE::verifySubIntEintOp(*this);
    }];
}

def NegEintOp : FHE_Op<"neg_eint"> {

    let summary = "Negates an encrypted integer";

    let arguments = (ins EncryptedIntegerType:$a);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a), [{
         build($_builder, $_state, a.getType(), a);
        }]>
    ];

    let verifier = [{
        return ::mlir::concretelang::FHE::verifyNegEintOp(*this);
    }];
}

def MulEintIntOp : FHE_Op<"mul_eint_int"> {

    let summary = "Mulitplies an encrypted integer and a clear integer";

    let arguments = (ins EncryptedIntegerType:$a, AnyInteger:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::concretelang::FHE::verifyMulEintIntOp(*this);
    }];
}

def ApplyLookupTableEintOp : FHE_Op<"apply_lookup_table"> {

    let summary = "Applies a clear lookup table to an encrypted integer";

    let arguments = (ins EncryptedIntegerType:$ct,
        TensorOf<[AnyInteger]>:$l_cst);
    let results = (outs EncryptedIntegerType);

    let verifier = [{
        return ::mlir::concretelang::FHE::verifyApplyLookupTable(*this);
    }];
}

#endif
