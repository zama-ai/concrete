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
include "concretelang/Dialect/GLWE/IR/GLWEAttrs.td"
include "mlir/IR/BuiltinTypes.td"

class GLWE_Type<string name, string mnemonic_, list<Trait> traits = []> :
  TypeDef<GLWE_Dialect, name, traits> {
      let mnemonic = mnemonic_;

      let assemblyFormat = "`<` struct(params) `>`";
  }

def GLWE_GLWE : GLWE_Type<"GLWE", "glwe"> {

  let summary = "Represents a GLWE ciphertext";

  let description = [{
    An type describing a GLWE ciphertext:
      - secret_key: The secret key encryption parameters
      - encoding: The encoding parameters
      - variance: An optional variance expression
  }];

  let parameters = (ins
    "GLWESecretKeyAttr": $secret_key,
    "GLWEEncodingAttr": $encoding,
    OptionalParameter<"GLWEExprAttr">: $variance
  );
}

def GLWE_RadixGLWE : GLWE_Type<"RadixGLWE", "radix_glwe"> {

  let summary = "Represents a decomposed GLWE ciphertext";

  let description = [{
    A type describing a decomposed GLWE ciphertext:"
      - secret_key: The secret key encryption parameters
      - encoding: The encoding parameters
      - decomp: Parameters of the decomposition
      - mask: Indicates if the mask is decomposed
      - body: Indicates if the body is decomposed
      - variance: An optional variance expression
  }];

  let parameters = (ins
    "GLWESecretKeyAttr":$secret_key,
    "GLWEEncodingAttr": $encoding,
    "DecompositionParametersAttr":$decomp,
    DefaultValuedParameter<"bool", "true">:$mask,
    DefaultValuedParameter<"bool", "true">:$body,
    OptionalParameter<"GLWEExprAttr">: $variance
  );

}

def GLWE_GLev : GLWE_Type<"GLev", "glev"> {

  let summary = "Represents a GLev ciphertext i.e. a collection of GLWE ciphertext";

  let description = [{
    A type describing a GLev ciphertext:"
      - secret_key: The secret key encryption parameters
      - encoding: The encoding parameters
      - decomp: Parameters of the decomposition
      - average_message:
      - shape: The shape of the GLWE ciphertext
      - variance: An optional variance expression
  }];

  let parameters = (ins
    "GLWESecretKeyAttr":$secret_key,
    "GLWEEncodingAttr": $encoding,
    "DecompositionParametersAttr":$decomp,
    "GLWEExprAttr": $average_message,
    ArrayRefParameter<"GLWEExprAttr">:$shape,
    OptionalParameter<"GLWEExprAttr">: $variance
  );

}

#endif
