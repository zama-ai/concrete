//===- GLWEOps.td - High level GLWE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWE_OPS
#define CONCRETELANG_DIALECT_GLWE_IR_GLWE_OPS

include "concretelang/Dialect/GLWE/IR/GLWEDialect.td"
include "concretelang/Dialect/GLWE/IR/GLWETypes.td"
include "concretelang/Dialect/GLWE/IR/GLWEAttrs.td"

class GLWE_Op<string mnemonic, list<Trait> traits = []> :
    Op<GLWE_Dialect, mnemonic, traits>;

def GLWE_ModulusSwitch: GLWE_Op<"modulus_switch"> {

    let summary = "Change the modulus of the a GLWE ciphertext.";

    let description = [{
        Change the mask and/or the body modulus of the input GLWE ciphertext by the one given in attribute:
            - input: The input ciphertext
            - modulus: The new modulus of the output ciphertext
            - mask: Optional attribute to indicate if the mask modulus is changed, default: true
            - body: Optional attribute to indicate if the mask modulus is changed, default: true

        The input and output GLWE types must share all their parameters unless the mask and body modulus depending of mask and body attributes.
    }];

    let arguments = (ins
        GLWE_GLWE:$input,
        GLWE_GLWEExprAttr: $modulus,
        DefaultValuedAttr<BoolAttr, "true">: $mask,
        DefaultValuedAttr<BoolAttr, "true">: $body,
        OptionalAttr<GLWE_GLWEExprAttr>: $variance
    );

    let results = (outs GLWE_GLWE:$output);

    let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";

    let hasVerifier = 1;
}

#endif
