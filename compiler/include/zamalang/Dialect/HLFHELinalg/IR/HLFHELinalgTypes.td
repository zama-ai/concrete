#ifndef ZAMALANG_DIALECT_HLFHELinalg_IR_HLFHELinalg_TYPES
#define ZAMALANG_DIALECT_HLFHELinalg_IR_HLFHELinalg_TYPES

include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.td"
include "mlir/IR/BuiltinTypes.td"
include "zamalang/Dialect/HLFHE/IR/HLFHETypes.td"

class HLFHELinalg_Type<string name, list<Trait> traits = []> :
  TypeDef<HLFHELinalg_Dialect, name, traits> { }

#endif
