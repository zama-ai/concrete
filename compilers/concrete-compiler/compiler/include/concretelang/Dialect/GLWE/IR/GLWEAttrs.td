//===- GLWEAttrs.td - GLWE dialect attrs ----------------*- tablegen
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWE_ATTRS
#define CONCRETELANG_DIALECT_GLWE_IR_GLWE_ATTRS

include "concretelang/Dialect/GLWE/IR/GLWEDialect.td"
include "concretelang/Dialect/GLWE/IR/GLWETypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

class GLWE_Attr<string name, string attrMnemonic, list<Trait> traits = []>: AttrDef<GLWE_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
    let assemblyFormat = "`<` struct(params) `>`";
}


def GLWE_ParameterVariableAttr: AttrDef<GLWE_Dialect, "Variable">{
    let mnemonic = "let";
    let description = [{
        A parameter with its domain definition:

          - name: name of the parameter
          - domain: the domain of the parameter
    }];

    let parameters = (ins
        "::mlir::concretelang::GLWE::Variable": $expr
    );
    let hasCustomAssemblyFormat = 1;
}

def GLWE_GlweExprAttr: AttrDef<GLWE_Dialect, "GlweExpr">{
    let mnemonic = "expr";


    let parameters = (ins
        "::mlir::concretelang::GLWE::GlweExpr": $expr
    );
    let hasCustomAssemblyFormat = 1;
}

def GLWE_GLWEParameters: GLWE_Attr<"GLWEParameters", "glwe_params"> {

    let description = [{
      An attribute describing GLWE parameters:

        - message_bound: maximal size of the message
        - dimension: the dimension of GLWE
        - mask_modulus: the modulus of the mask
        - body_modulus: the modulus of the body
        - polynomial_size: The size of the polynomial
    }];

    let parameters = (ins
        "unsigned": $message_bound,
        GLWE_GlweExprAttr: $dimension,
        GLWE_GlweExprAttr: $poly_size,
        "unsigned": $mask_modulus,
        "unsigned": $body_modulus,
        OptionalParameter<"unsigned">: $variance
    );

}

def GLWE_DecompositionParameters: GLWE_Attr<"DecompositionParameters", "decomp_params"> {

    let description = [{
      An attribute describing GLWE decomposition parameters:

        - base_log: log2 of the decomposition base
        - level: the number of level of the decomposition
    }];

    let parameters = (ins
        "unsigned": $base_log,
        "unsigned": $level
    );

}

#endif
