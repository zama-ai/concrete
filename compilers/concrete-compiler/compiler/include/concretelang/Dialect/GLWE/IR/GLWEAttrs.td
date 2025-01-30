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

def GLWE_GlweExprAttr: AttrDef<GLWE_Dialect, "GlweExpr">{
    let mnemonic = "expr";

    let parameters = (ins
        "::mlir::concretelang::GLWE::GlweExpr": $expr
    );
    let hasCustomAssemblyFormat = 1;
}

#endif
