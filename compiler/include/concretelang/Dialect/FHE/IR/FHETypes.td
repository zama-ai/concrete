#ifndef CONCRETELANG_DIALECT_FHE_IR_FHE_TYPES
#define CONCRETELANG_DIALECT_FHE_IR_FHE_TYPES

include "concretelang/Dialect/FHE/IR/FHEDialect.td"
include "mlir/IR/BuiltinTypes.td"

class FHE_Type<string name, list<Trait> traits = []> :
  TypeDef<FHE_Dialect, name, traits> { }

def FHE_EncryptedIntegerType : FHE_Type<"EncryptedInteger",
        [MemRefElementTypeInterface]> {
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
}

#endif
