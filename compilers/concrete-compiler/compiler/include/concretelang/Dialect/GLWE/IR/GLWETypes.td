//===- GLWETypes.td - GLWE dialect types ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWE_TYPES
#define CONCRETELANG_DIALECT_GLWE_IR_GLWE_TYPES

include "concretelang/Dialect/GLWE/IR/GLWEDialect.td"
// include "concretelang/Dialect/GLWE/Interfaces/GLWEInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

class GLWE_Type<string name, string mnemonic_, list<Trait> traits = []> :
  TypeDef<GLWE_Dialect, name, traits> { 
      let mnemonic = mnemonic_;

      let assemblyFormat = "`<` struct(params) `>`";
  }

def GLWE_GLWE : GLWE_Type<"GLWE", "glwe"> {

  let summary = "Represents GLWE ciphertext";

  let parameters = (ins
    "GLWEParametersAttr":$params
  );

}

def GLWE_RadixGLWE : GLWE_Type<"RadixGLWE", "radix_glwe"> {

  let summary = "Represents a decomposed GLWE ciphertext";

  let parameters = (ins
    "GLWEParametersAttr":$params,
    "DecompositionParametersAttr":$decomp,
    DefaultValuedParameter<"bool", "false">:$partial
  );

}

def GLWE_GLev : GLWE_Type<"GLev", "glev"> {

  let summary = "Represents a GLev ciphertext i.e. a collection of GLWE ciphertext";

  let parameters = (ins
    "GLWEParametersAttr":$params,
    "DecompositionParametersAttr":$decomp,
    "unsigned":$size,
    DefaultValuedParameter<"bool", "false">:$partial
  );

}

#endif
