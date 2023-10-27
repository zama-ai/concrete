//===- TFHEAttrs.td - TFHE dialect attrs ----------------*- tablegen
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_TFHE_IR_TFHE_ATTRS
#define CONCRETELANG_DIALECT_TFHE_IR_TFHE_ATTRS

include "concretelang/Dialect/TFHE/IR/TFHEDialect.td"
include "concretelang/Dialect/TFHE/IR/TFHETypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

class TFHE_Attr<string name, string attrMnemonic, list<Trait> traits = []>: AttrDef<TFHE_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

def TFHE_KeyswitchKeyAttr: TFHE_Attr<"GLWEKeyswitchKey", "ksk"> {

    let summary = "An attribute representing keyswitch key.";

    let parameters = (ins
        "mlir::concretelang::TFHE::GLWESecretKey":$inputKey,
        "mlir::concretelang::TFHE::GLWESecretKey":$outputKey,
        "int":$levels,
        "int":$baseLog,
        DefaultValuedParameter<"int", "-1">: $index
    );

    let assemblyFormat = " (`[` $index^ `]`)? `<` $inputKey `,` $outputKey `,` $levels `,` $baseLog `>`";
}

def TFHE_BootstrapKeyAttr: TFHE_Attr<"GLWEBootstrapKey", "bsk"> {

    let summary = "An attribute representing bootstrap key.";

    let parameters = (ins
        "mlir::concretelang::TFHE::GLWESecretKey":$inputKey,
        "mlir::concretelang::TFHE::GLWESecretKey":$outputKey,
        "int":$polySize,
        "int":$glweDim,
        "int":$levels,
        "int":$baseLog,
        DefaultValuedParameter<"int", "-1">: $index
    );

    let assemblyFormat = "(`[` $index^ `]`)? `<` $inputKey `,` $outputKey `,` $polySize `,` $glweDim `,` $levels `,` $baseLog `>`";
}

def TFHE_PackingKeyswitchKeyAttr: TFHE_Attr<"GLWEPackingKeyswitchKey", "pksk"> {

    let summary = "An attribute representing Wop Pbs key.";

    let parameters = (ins
        "mlir::concretelang::TFHE::GLWESecretKey":$inputKey,
        "mlir::concretelang::TFHE::GLWESecretKey":$outputKey,
        "int" : $outputPolySize,
        "int" : $innerLweDim,
        "int" : $glweDim,
        "int" : $levels,
        "int" : $baseLog,
        DefaultValuedParameter<"int", "-1">: $index
);

let assemblyFormat = " (`[` $index^ `]` )? `<` $inputKey `,` $outputKey`,` $outputPolySize`,` $innerLweDim `,` $glweDim `,` $levels `,` $baseLog `>`";

}

#endif
