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


def ApplyUnivariateRegion : Region<
  CPred<"::mlir::zamalang::predApplyUnivariateRegion($_self)">,
  "apply_univariate region needs one block with one any integer argument">;


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
    let arguments = (ins EncryptedIntegerType:$x);
    let results = (outs EncryptedIntegerType:$result);

    let builders = [
        OpBuilder<(ins "Value": $x), [{
         build($_builder, $_state, x.getType(), x);
        }]>
    ];


    let regions = (region ApplyUnivariateRegion:$body);
    // let assemblyFormat = "$x `:` type($x) $body attr-dict `:` type($result)";
}


def ReturnOp : HLFHE_Op<"apply_univariate_return", [NoSideEffect, ReturnLike, Terminator]> {
  let summary = "terminator of apply_univariate block";
  let arguments = (ins AnyInteger);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}



#endif
