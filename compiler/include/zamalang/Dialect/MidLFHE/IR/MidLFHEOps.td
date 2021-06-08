//===- MidLFHEOps.td - High level FHE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_OPS
#define ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.td"
include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.td"

class MidLFHE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MidLFHE_Dialect, mnemonic, traits>;

def AddPlainOp : MidLFHE_Op<"add_plain"> {
    let arguments = (ins GLWECipherTextType:$a, AnyInteger:$b);
    let results = (outs GLWECipherTextType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
            build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::zamalang::verifyAddPlainOp(*this);
    }];
}

def MulPlainOp : MidLFHE_Op<"mul_plain"> {
    let arguments = (ins GLWECipherTextType:$a, AnyInteger:$b);
    let results = (outs GLWECipherTextType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
            build($_builder, $_state, a.getType(), a, b);
        }]>
    ];
}

def HAddOp : MidLFHE_Op<"h_add"> {
    let arguments = (ins GLWECipherTextType:$a, GLWECipherTextType:$b);
    let results = (outs GLWECipherTextType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
            build($_builder, $_state, a.getType(), a, b);
        }]>
    ];
}


def HMulOp : MidLFHE_Op<"h_mul"> {
    let arguments = (ins GLWECipherTextType:$a, GLWECipherTextType:$b);
    let results = (outs GLWECipherTextType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
            build($_builder, $_state, a.getType(), a, b);
        }]>
    ];
}


def PBSRegion : Region<
  CPred<"::mlir::zamalang::predPBSRegion($_self)">,
  "pbs region needs one block with one any integer argument">;

def PBSOp : MidLFHE_Op<"pbs"> {
  let arguments = (ins GLWECipherTextType:$x, I32Attr:$big_n, I32Attr:$log_noise, I32Attr:$base_log, I32Attr:$level);
  let results = (outs GLWECipherTextType:$result);
  let regions = (region PBSRegion:$region);
}

def ReturnOp : MidLFHE_Op<"pbs_return", [NoSideEffect, ReturnLike, Terminator]> {
  let summary = "terminator of pbs block";
  let arguments = (ins AnyInteger);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

#endif
