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

def GLWE_GLWEExprAttr: AttrDef<GLWE_Dialect, "GLWEExpr">{
    let mnemonic = "expr";

    let parameters = (ins
        "::mlir::concretelang::GLWE::GLWEExpr": $expr
    );
    let hasCustomAssemblyFormat = 1;
}

def GLWE_GLWEParametersAttr: GLWE_Attr<"GLWEParameters", "glwe_params"> {

  let description = [{
    An attribute describing GLWE parameters :
      - dimension: the dimension of GLWE
      - mask_modulus: the modulus of the mask
      - body_modulus: the modulus of the body
      - polynomial_size: The size of the polynomial
  }];

  let parameters = (ins
      GLWE_GLWEExprAttr: $dimension,
      GLWE_GLWEExprAttr: $poly_size,
      GLWE_GLWEExprAttr: $mask_modulus,
      GLWE_GLWEExprAttr: $body_modulus
  );

}

def GLWE_DecompositionParameters: GLWE_Attr<"DecompositionParameters", "decomp"> {

    let description = [{
      An attribute describing decomposition parameters:
        - base_log: log2 of the decomposition base
        - level: the number of level of the decomposition
    }];

    let parameters = (ins
        GLWE_GLWEExprAttr: $base_log,
        GLWE_GLWEExprAttr: $level
    );

}

#endif
