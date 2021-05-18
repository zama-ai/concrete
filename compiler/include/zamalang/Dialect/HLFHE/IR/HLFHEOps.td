//===- HLFHEOps.td - High level FHE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZAMALANG_DIALECT_HLFHE_IR_HLFHE_OPS
#define ZAMALANG_DIALECT_HLFHE_IR_HLFHE_OPS

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

def NegEintOp : HLFHE_Op<"neg_eint"> {
    let arguments = (ins EncryptedIntegerType:$a);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a), [{
         build($_builder, $_state, a.getType(), a);
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

def MulEintOp : HLFHE_Op<"mul_eint"> {
    let arguments = (ins EncryptedIntegerType:$a, EncryptedIntegerType:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];
}

def ApplyUnivariateOp : HLFHE_Op<"apply_univariate"> {
    // TODO: express a functionLike?
    let arguments = (ins EncryptedIntegerType:$a);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a), [{
         build($_builder, $_state, a.getType(), a);
        }]>
    ];
}


#endif
