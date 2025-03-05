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
include "concretelang/Dialect/GLWE/IR/GLWEInterfaces.td"

class GLWE_Op<string mnemonic, list<Trait> traits = []> :
    Op<GLWE_Dialect, mnemonic, traits>;

def GLWE_GenericOP: GLWE_Op<"generic", [DeclareOpInterfaceMethods<GLWEOpInterface>]> {
    let summary = "Generic GLWE operator";

    let arguments = (ins
        Variadic<GLWETypeInterface>:$in,
        OptionalAttr<GLWE_GLWEExprAttr>: $variance
    );

    let results = (outs GLWETypeInterface:$out);

    let assemblyFormat = "operands attr-dict `:` type($in) `->` type($out)";

}

def GLWE_Dot: GLWE_Op<"dot", [DeclareOpInterfaceMethods<GLWEOpInterface>]> {
    let summary = "Linear combination of cyphertext with clear coefficient.";

    let arguments = (ins
        1DTensorOf<[GLWE_GLWE]>:$input,
        ArrayAttr:$weights,
        OptionalAttr<GLWE_GLWEExprAttr>: $variance
    );

    let results = (outs GLWE_GLWE:$output);

    let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";

    let hasVerifier = 1;

}

def GLWE_ModulusSwitch: GLWE_Op<"modulus_switch", [DeclareOpInterfaceMethods<GLWEOpInterface>]> {

    let summary = "Change the modulus of the a GLWE ciphertext.";

    let description = [{
        Change the mask and/or the body modulus of the input GLWE ciphertext by the one given in attribute:
            - input: The input ciphertext
            - modulus: The new modulus of the output ciphertext
            - mask: Optional attribute to indicate if the mask modulus is changed, default: true
            - body: Optional attribute to indicate if the body modulus is changed, default: true

        The input and output GLWE types must share all their parameters unless the mask and body modulus depending of mask and body attributes.
    }];

    let arguments = (ins
        GLWE_GLWE:$input,
        GLWE_GLWEExprAttr: $modulus,
        DefaultValuedAttr<BoolAttr, "true">: $mask,
        DefaultValuedAttr<BoolAttr, "true">: $body,
        OptionalAttr<GLWE_GLWEExprAttr>: $variance,
        OptionalAttr<GLWE_GLWEExprAttr>: $cost
    );

    let results = (outs GLWE_GLWE:$output);

    let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";

    let hasVerifier = 1;
}

def GLWE_ExactDecompose: GLWE_Op<"exact_decompose", [DeclareOpInterfaceMethods<GLWEOpInterface>]> {

    let summary = "Decompose a GLWE ciphertext in `level` elements in a given `base`.";

    let description = [{
        Change the mask and/or the body modulus of the input GLWE ciphertext by the one given in attribute:
            - input: The input ciphertext
            - decomposition: The decomposition to apply
            - mask: Optional attribute to indicate if the mask is decomposed, default: true
            - body: Optional attribute to indicate if the body is decomposed, default: true

        The input GWLE and output RadixGLWE types must share all their parameters.
        The output RadixGLWE must have the same decomposition, mask and body parameter that the decomposition, mask and body parameter of the op.
    }];

    let arguments = (ins
        GLWE_GLWE:$input,
        GLWE_DecompositionParametersAttr: $decomposition,
        DefaultValuedAttr<BoolAttr, "true">: $mask,
        DefaultValuedAttr<BoolAttr, "true">: $body,
        OptionalAttr<GLWE_GLWEExprAttr>: $variance
    );

    let results = (outs GLWE_RadixGLWE:$output);

    let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";

    let hasVerifier = 1;
}

def GLWE_ExactRecompose: GLWE_Op<"exact_recompose", [DeclareOpInterfaceMethods<GLWEOpInterface>]> {

    let summary = "Recompose a RadiwGLWE ciphertext using a glev.";

    let description = [{
        Change the mask and/or the body modulus of the input GLWE ciphertext by the one given in attribute:
            - input: The input ciphertext
            - glev: The glev used for the recomposition
            - grouping_factor: TODO
            - mantissa_size: TODO

        The polynomial size of the input GWLE secret key + the number of bodies of the input ciphertext
        (i.e. if body == true its the nb_keys of the input else 0) must be equal to the size of the glev.
        The input RadixGLWE and the GLev must have the same decomposition parameter.
        mask and body parameter of the op.
    }];

    let arguments = (ins
        GLWE_RadixGLWE:$input,
        GLWE_GLev:$glev,
        // TODO - GLWE_GLWEExprAttr
        DefaultValuedAttr<I64Attr, "1">:$grouping_factor,
        DefaultValuedAttr<I64Attr, "53">: $mantissa_size,
        DefaultValuedAttr<BoolAttr, "false">: $fft,
        OptionalAttr<GLWE_GLWEExprAttr>: $variance
    );

    let results = (outs GLWE_GLWE:$output);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

    let hasVerifier = 1;
}

def GLWE_SampleExtract: GLWE_Op<"sample_extract", [DeclareOpInterfaceMethods<GLWEOpInterface>]> {

    let summary = "Extract a polynomial encrypted coefficient.";

    let description = [{
        Extract a polynomial encrypted coefficient.
            - input: The input ciphertext

        Verify that:
            - The ouput secret key must have poly_size equal to 1 and dimension equal to input secret key dimension * poly_size.
            - The input and output encoding must have the same parameters.

        The default output variance is equals to the input variance.
    }];

    let arguments = (ins
        GLWE_GLWE:$input,
        OptionalAttr<GLWE_GLWEExprAttr>: $variance
    );

    let results = (outs GLWE_GLWE:$output);

    let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";

    let hasVerifier = 1;
}

#endif
