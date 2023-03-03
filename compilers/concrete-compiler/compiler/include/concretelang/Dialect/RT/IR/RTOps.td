#ifndef CONCRETELANG_DIALECT_RT_IR_RT_OPS
#define CONCRETELANG_DIALECT_RT_IR_RT_OPS

include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

include "concretelang/Dialect/RT/IR/RTDialect.td"
include "concretelang/Dialect/RT/IR/RTTypes.td"

class RT_Op<string mnemonic, list<Trait> traits = []> :
    Op<RT_Dialect, mnemonic, traits>;

def RT_DataflowTaskOp : RT_Op<"dataflow_task", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<AllocationOpInterface,
       ["buildDealloc", "buildClone"]>,
    SingleBlockImplicitTerminator<"DataflowYieldOp">,
    AutomaticAllocationScope] > {
    let arguments = (ins Variadic<AnyType>: $inputs);
    let results = (outs Variadic<AnyType>: $outputs);

    let regions = (region AnyRegion:$body);

    let builders = [
      OpBuilder<(ins
        CArg<"TypeRange", "{}">: $resultTypes,
        CArg<"ValueRange", "{}">: $operands,
        CArg<"ArrayRef<NamedAttribute>", "{}">: $attrs)>
    ];
    let skipDefaultBuilders = 1;

    let summary = "Dataflow task operation";
    let description = [{

`RT.dataflow_task` allows to specify a task that will be concurrently
executed when their operands are ready.  Operands are either the
results of computation in other `RT.dataflow_task` (dataflow
dependences) or obtained from the execution context (immediate
operands).  Operands are synchronized using futures and, in the case
of immediate operands, copied when the task is created.  Caution is
required when the operand is a pointer as no deep copy will occur.

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
```
    }];
}

def RT_DataflowYieldOp : RT_Op<"dataflow_yield", [ReturnLike, Terminator]> {
    let arguments = (ins Variadic<AnyType>: $values);

    let summary = "Dataflow yield operation";
    let description = [{
`RT.dataflow_yield` is a special terminator operation for blocks inside the region
in `RT.dataflow_task`. It allows to specify the return values of a `RT.dataflow_task`.

Example:

```mlir
%0 = constant 1 : i64
%1 = constant 2 : i64
"RT.dataflow_yield" %0, %1 : i64, i64
```
  }];
}

def RT_MakeReadyFutureOp : RT_Op<"make_ready_future", [
    		      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    		      DeclareOpInterfaceMethods<AllocationOpInterface,
       		      ["buildDealloc", "buildClone"]>]> {
    let arguments = (ins AnyType: $input,
			 AnyType: $memrefCloned);
    let results = (outs RT_Future: $output);
    let summary = "Build a ready future.";
    let description = [{
Data passed to dataflow tasks must be encapsulated in futures,
including immediate operands.  These must be converted into futures
using `RT.make_ready_future`.
}];
}

def RT_AwaitFutureOp : RT_Op<"await_future"> {
    let arguments = (ins RT_Future: $input);
    let results = (outs AnyType: $output);
    let summary = "Wait for a future and access its data.";
    let description = [{
The results of a dataflow task are always futures which could be
further used as inputs to subsequent tasks.  When the result of a task
is needed in the outer execution context, the result future needs to
be synchronized and its data accessed using `RT.await_future`.
}];
}

def RT_CreateAsyncTaskOp : RT_Op<"create_async_task"> {
    let arguments = (ins SymbolRefAttr:$workfn,
    		    	 Variadic<AnyType>:$list);
    let results = (outs );
    let summary = "Create a dataflow task.";
}

def RT_RegisterTaskWorkFunctionOp : RT_Op<"register_task_work_function"> {
    let arguments = (ins Variadic<AnyType>:$list);
    let results = (outs );
    let summary = "Register the task work-function with the runtime system.";
}

def RT_CloneFutureOp : RT_Op<"clone_future",
     [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<AllocationOpInterface,
       ["buildDealloc", "buildClone"]>] > {
    let builders = [
      OpBuilder<(ins "Value": $input), [{
      return build($_builder, $_state, input.getType(), input);
    }]>];

    let arguments = (ins RT_Future: $input);
    let results = (outs RT_Future: $output);
}

def RT_DeallocateFutureOp : RT_Op<"deallocate_future"> {
    let arguments = (ins AnyType: $input);
    let results = (outs );
}

def RT_DeallocateFutureDataOp : RT_Op<"deallocate_future_data"> {
    let arguments = (ins RT_Future: $input);
    let results = (outs );
}

def RT_BuildReturnPtrPlaceholderOp : RT_Op<"build_return_ptr_placeholder"> {
    let arguments = (ins );
    let results = (outs RT_Pointer: $output);
}

def RT_DerefReturnPtrPlaceholderOp : RT_Op<"deref_return_ptr_placeholder"> {
    let arguments = (ins RT_Pointer: $input);
    let results = (outs RT_Future: $output);
}

def RT_DerefWorkFunctionArgumentPtrPlaceholderOp : RT_Op<"deref_work_function_argument_ptr_placeholder"> {
    let arguments = (ins RT_Pointer: $input);
    let results = (outs AnyType: $output);
}

def RT_WorkFunctionReturnOp : RT_Op<"work_function_return"> {
    let arguments = (ins AnyType:$in, AnyType:$out);
    let results = (outs );
}

#endif
