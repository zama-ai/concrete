#ifndef CONCRETELANG_DIALECT_TYPEINFERENCE_IR_TYPEINFERENCE_OPS
#define CONCRETELANG_DIALECT_TYPEINFERENCE_IR_TYPEINFERENCE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "concretelang/Dialect/TypeInference/IR/TypeInferenceDialect.td"
include "concretelang/Dialect/FHE/IR/FHETypes.td"
include "concretelang/Dialect/TFHE/IR/TFHETypes.td"
include "concretelang/Dialect/Concrete/IR/ConcreteTypes.td"

class TypeInference_Op<string mnemonic, list<Trait> traits = []>
    : Op<TypeInference_Dialect, mnemonic, traits>;

def TypeInference_PropagateDownwardOp : TypeInference_Op<"propagate_downward"> {
  let summary = "Causes the type of the operand to be forwarded upon type inference.";

  let arguments = (ins AnyType);
  let results = (outs AnyType);
}

def TypeInference_PropagateUpwardOp : TypeInference_Op<"propagate_upward"> {
  let summary = "Causes the type of the result to be forwarded to the operands upon type inference.";

  let arguments = (ins AnyType);
  let results = (outs AnyType);
}

def TypeInference_UnresolvedConflictOp : TypeInference_Op<"unresolved_conflict"> {
  let summary = "Represents an unresolved conflict";

  let arguments = (ins AnyType);
  let results = (outs AnyType);
}

#endif
