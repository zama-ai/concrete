#ifndef ZAMALANG_DIALECT_RT_IR_RT_OPS
#define ZAMALANG_DIALECT_RT_IR_RT_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "zamalang/Dialect/RT/IR/RTDialect.td"
include "zamalang/Dialect/RT/IR/RTTypes.td"

class RT_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<RT_Dialect, mnemonic, traits>;

def DataflowTaskOp : RT_Op<"dataflow_task", [SingleBlockImplicitTerminator<"DataflowYieldOp">]> {
    let arguments = (ins Variadic<AnyType>: $inputs);
    let results = (outs Variadic<AnyType>:$outputs);

    let regions = (region AnyRegion:$body);

    let summary = "Dataflow task operation";
    let description = [{
`RT.dataflow_task` allows to specify a task that will be concurrently executed when their operands are ready.

Example:

```mlir
func @test(%0 : i64): (i64, i64) {
    // Execute right now as %0 is ready.
    %1, %2 = "RT.dataflow_task"(%0) ({
        %a = addi %0, %0 : i64
        %b = muli %0, %0 : i64
        "RT.dataflow_yield"(%a, %b) : (i64, i64) -> i64
    }) : (i64, i64) -> (i64, i64)
    // Concurrently execute both tasks below when the task above is completed.
    %3 = "RT.dataflow_task"(%1) ({
        %c = constant 1 : %i64
        %a = addi %1, %c : i64
        "RT.dataflow_yield"(%a) : (i64, i64) -> i64
    }) : (i64, i64) -> (i64, i64)
    %4 = "RT.dataflow_task"(%2) ({
        %c = constant 2 : %i64
        %a = addi %2, %c : i64
        "RT.dataflow_yield"(%a) : (i64, i64) -> i64
    }) : (i64, i64) -> (i64, i64)
    return %3, %4 : (i64, i64)
}
    }];
}

def DataflowYieldOp : RT_Op<"dataflow_yield", [ReturnLike, Terminator]> {
    let arguments = (ins Variadic<AnyType>: $values);

    let summary = "Dataflow yield operation";
    let description = [{
`RT.dataflow_yield` is a special terminator operation for blocks inside the region
in `RT.dataflow_task`. It allows to specify the returns values of a `RT.dataflow_task`.

Example:

```mlir
%0 = constant 1 : i64
%1 = constant 2 : i64
"RT.dataflow_yield" %0, %1 : i64, i64
```
  }];
}

#endif
