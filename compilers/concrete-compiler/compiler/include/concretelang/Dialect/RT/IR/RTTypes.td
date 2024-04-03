#ifndef CONCRETELANG_DIALECT_FHE_IR_FHE_TYPES
#define CONCRETELANG_DIALECT_FHE_IR_FHE_TYPES

include "concretelang/Dialect/RT/IR/RTDialect.td"
include "mlir/IR/BuiltinTypes.td"

class RT_Type<string name, list<Trait> traits = []> :
  TypeDef<RT_Dialect, name, traits> { }

def RT_Future : RT_Type<"Future", [MemRefElementTypeInterface]> {
  let mnemonic = "future";

  let summary = "Future with a parameterized element type";

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

  let hasCustomAssemblyFormat = 1;
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

  let hasCustomAssemblyFormat = 1;
}

#endif
