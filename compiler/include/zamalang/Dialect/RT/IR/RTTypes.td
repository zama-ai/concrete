#ifndef ZAMALANG_DIALECT_HLFHE_IR_HLFHE_TYPES
#define ZAMALANG_DIALECT_HLFHE_IR_HLFHE_TYPES

include "zamalang/Dialect/RT/IR/RTDialect.td"
include "mlir/IR/BuiltinTypes.td"

class RT_Type<string name, list<Trait> traits = []> :
  TypeDef<RT_Dialect, name, traits> { }

def RT_Future : RT_Type<"Future"> {
  let mnemonic = "future";

  let summary = "future with a parameterized element type";

  let description = [{
    The value of a `!RT.future` type represents the result of an asynchronous operation.

    Examples:

    ```mlir
    !RT.future<i64>
    ```
  }];

  let parameters = (ins "Type":$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];

  let printer = [{
    $_printer << "future<";
    $_printer.printType(getElementType());
    $_printer << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    Type elementType;
    if ($_parser.parseType(elementType))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, elementType);
  }];
}

def RT_Pointer : RT_Type<"Pointer"> {
  let mnemonic = "rtptr";

  let summary = "Pointer to a parameterized element type";

  let description = [{
  }];

  let parameters = (ins "Type":$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];

  let printer = [{
    $_printer << "rtptr<";
    $_printer.printType(getElementType());
    $_printer << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    Type elementType;
    if ($_parser.parseType(elementType))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, elementType);
  }];
}

#endif
