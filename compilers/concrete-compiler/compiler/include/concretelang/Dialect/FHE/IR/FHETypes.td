//===- FHETypes.td - FHE dialect types ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_FHE_IR_FHE_TYPES
#define CONCRETELANG_DIALECT_FHE_IR_FHE_TYPES

include "concretelang/Dialect/FHE/IR/FHEDialect.td"
include "concretelang/Dialect/FHE/Interfaces/FHEInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

class FHE_Type<string name, list<Trait> traits = []> :
  TypeDef<FHE_Dialect, name, traits> { }

def FHE_EncryptedUnsignedIntegerType : FHE_Type<"EncryptedUnsignedInteger",
        [MemRefElementTypeInterface, FheIntegerInterface]> {
  let mnemonic = "eint";

  let summary = "An encrypted unsigned integer";

  let description = [{
    An encrypted unsigned integer with `width` bits to performs FHE Operations.

    Examples:
    ```mlir
    !FHE.eint<7>
    !FHE.eint<6>
    ```
  }];

  let parameters = (ins "unsigned":$width);

  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = true;

  let extraClassDeclaration = [{
    bool isSigned() const { return false; }
    bool isUnsigned() const { return true; }
  }];
}

def FHE_EncryptedSignedIntegerType : FHE_Type<"EncryptedSignedInteger",
        [MemRefElementTypeInterface, FheIntegerInterface]> {
  let mnemonic = "esint";

  let summary = "An encrypted signed integer";

  let description = [{
    An encrypted signed integer with `width` bits to performs FHE Operations.

    Examples:
    ```mlir
    !FHE.esint<7>
    !FHE.esint<6>
    ```
  }];

  let parameters = (ins "unsigned":$width);

  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = true;

  let extraClassDeclaration = [{
    bool isSigned() const { return true; }
    bool isUnsigned() const { return false; }
  }];
}

def FHE_AnyEncryptedInteger : Type<Or<[
  FHE_EncryptedUnsignedIntegerType.predicate,
  FHE_EncryptedSignedIntegerType.predicate
]>>;

def FHE_EncryptedBooleanType : FHE_Type<"EncryptedBoolean",
        [MemRefElementTypeInterface]> {
  let mnemonic = "ebool";

  let summary = "An encrypted boolean";

  let description = [{
    An encrypted boolean.
  }];

  let extraClassDeclaration = [{
    /// Returns the required number of bits to represent an encrypted boolean
    static size_t getWidth() { return 2; }
  }];
}

#endif
