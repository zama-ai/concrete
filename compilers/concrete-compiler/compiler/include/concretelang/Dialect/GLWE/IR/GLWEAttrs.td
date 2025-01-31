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

def GLWE_GLWESizeAttr: GLWE_Attr<"GLWESize", "size"> {
    let summary = "Represent the size of a GLWE.";

    let description = [{
        - dimension: The dimension of GLWE
        - poly_size: The size of the polynom
    }];

    let parameters = (ins
        GLWE_GLWEExprAttr: $dimension,
        GLWE_GLWEExprAttr: $poly_size
    );
}

def GLWE_GLWESecretKeyDistributionAttr: GLWE_Attr<"GLWESecretKeDistribution", "distribution"> {

    let summary = "Represent distribution of a secret key.";

    let parameters = (ins
        GLWE_GLWEExprAttr: $average_mean,
        GLWE_GLWEExprAttr: $average_variance
    );

}

def GLWE_GLWESecretKeyAttr: GLWE_Attr<"GLWESecretKey", "secret_key"> {

    let summary = "Represent secret key parameters.";

    let description = [{
        - size: The size of the secret key
        - distribution: The distribution of the secret key
    }];

    let parameters = (ins
        GLWE_GLWESizeAttr: $size,
        GLWE_GLWESecretKeyDistributionAttr: $distribution
    );

}

def GLWE_GLWEEncodingAttr: GLWE_Attr<"GLWEEncoding", "encoding"> {
    let summary = "Represent encoding of a GLWE.";

    let description = [{
        - body_modulus: The encoding modulus of the body
        - mask_modulus: The encoding modulus of the mask
        - message_modulus: Parameter
        - right_padding: The number of bit of the right padding
        - left_padding: The number of bit of the left padding
    }];

    let parameters = (ins
        GLWE_GLWEExprAttr: $body_modulus,
        GLWE_GLWEExprAttr: $mask_modulus,
        GLWE_GLWEExprAttr: $message_modulus,
        GLWE_GLWEExprAttr: $right_padding,
        GLWE_GLWEExprAttr: $left_padding
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
