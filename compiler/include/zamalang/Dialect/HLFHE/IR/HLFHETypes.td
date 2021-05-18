#ifndef ZAMALANG_DIALECT_HLFHE_IR_HLFHE_TYPES
#define ZAMALANG_DIALECT_HLFHE_IR_HLFHE_TYPES

include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.td"

class HLFHE_Type<string name> : TypeDef<HLFHE_Dialect, name> { }

def EncryptedIntegerType : HLFHE_Type<"EncryptedInteger"> {
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
    return get($_ctxt, width);
  }];
}

#endif
