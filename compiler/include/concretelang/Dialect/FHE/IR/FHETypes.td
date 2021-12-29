#ifndef CONCRETELANG_DIALECT_FHE_IR_FHE_TYPES
#define CONCRETELANG_DIALECT_FHE_IR_FHE_TYPES

include "concretelang/Dialect/FHE/IR/FHEDialect.td"
include "mlir/IR/BuiltinTypes.td"

class FHE_Type<string name, list<Trait> traits = []> :
  TypeDef<FHE_Dialect, name, traits> { }

def EncryptedIntegerType : FHE_Type<"EncryptedInteger",
        [MemRefElementTypeInterface]> {
  let mnemonic = "eint";

  let summary = "An encrypted integer";

  let description = [{
    An encrypted integer with clear precision of width.
  }];

  let parameters = (ins "unsigned":$width);

  // We define the printer inline.
  let printer = [{
    $_printer << "eint<" << getImpl()->width << ">";
  }];

  // The parser is defined here also.
  let parser = [{
    if ($_parser.parseLess())
      return Type();
    int width;
    if ($_parser.parseInteger(width))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    Location loc = $_parser.getEncodedSourceLoc($_parser.getNameLoc());
    return getChecked(loc, loc.getContext(), width);
  }];

  let genVerifyDecl = true;
}

#endif
