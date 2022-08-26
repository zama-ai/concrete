#ifndef CONCRETELANG_DIALECT_FHE_IR_FHE_TYPES
#define CONCRETELANG_DIALECT_FHE_IR_FHE_TYPES

include "concretelang/Dialect/FHE/IR/FHEDialect.td"
include "concretelang/Dialect/FHE/IR/FHEInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

class FHE_Type<string name, list<Trait> traits = []> :
  TypeDef<FHE_Dialect, name, traits> { }

def FHE_EncryptedIntegerType : FHE_Type<"EncryptedInteger",
        [MemRefElementTypeInterface, FheIntegerInterface]> {
  let mnemonic = "eint";

  let summary = "An encrypted integer";

  let description = [{
    An encrypted integer with `width` bits to performs FHE Operations.

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
  FHE_EncryptedIntegerType.predicate,
  FHE_EncryptedSignedIntegerType.predicate
]>>;

#endif
