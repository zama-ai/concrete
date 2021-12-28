#ifndef CONCRETELANG_DIALECT_RT_ANALYSIS_AUTOPAR
#define CONCRETELANG_DIALECT_RT_ANALYSIS_AUTOPAR

include "mlir/Pass/PassBase.td"

def BuildDataflowTaskGraph : Pass<"BuildDataflowTaskGraph", "mlir::ModuleOp"> {
  let summary =
      "Identify profitable dataflow tasks and build DataflowTaskGraph.";

  let description = [{
  This pass builds a dataflow graph out of a HLFHE program.

  In its current incarnation, it considers some heavier weight
  operations (e.g., HLFHELinalg Dot and Matmult or bootstraps) as
  candidates for being executed in a discrete task, and then
  sinks within the task the lighter weight operation that do not
  increase the graph cut (amount of dependences in or out).

  The output is a program partitioned in RT::DataflowTaskOp that
  expose task dependences as arguments and results of the
  DataflowTaskOp.

  Example:

```mlir
  func @main(%arg0: tensor<3x4x!HLFHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>> {
    %0 = "HLFHELinalg.matmul_eint_int"(%arg0, %arg1) : (tensor<3x4x!HLFHE.eint<2>>, tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
    return %0 : tensor<3x2x!HLFHE.eint<2>>
  }
```

  Will result in generating a dataflow task for the Matmul operation:

```mlir
  func @main(%arg0: tensor<3x4x!HLFHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>> {
    %0 = "RT.dataflow_task"(%arg0, %arg1) ( {
      %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %arg1) : (tensor<3x4x!HLFHE.eint<2>>, tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
      "RT.dataflow_yield"(%1) : (tensor<3x2x!HLFHE.eint<2>>) -> ()
    }) : (tensor<3x4x!HLFHE.eint<2>>, tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
    return %0 : tensor<3x2x!HLFHE.eint<2>>
  }
```
  }];
}

def BufferizeDataflowTaskOps : Pass<"BufferizeDataflowTaskOps", "mlir::ModuleOp"> {
  let summary =
      "Bufferize DataflowTaskOp(s).";

  let description = [{
  This pass lowers DataflowTaskOp arguments and results from tensors
  to mlir::memref. It also lowers the arguments of DataflowYieldOp.
  }];
}

def FixupDataflowTaskOps : Pass<"FixupDataflowTaskOps", "mlir::ModuleOp"> {
  let summary =
      "Fix DataflowTaskOp(s) before lowering.";

  let description = [{
  This pass fixes up code changes that intervene between the
  BuildDataflowTaskGraph pass and the lowering of the taskgraph to
  LLVMIR and calls to the DFR runtime system.

  In particular, some operations (e.g., constants, dimension
  operations, etc.) can be used within the task while only defined
  outside. In most cases cloning and sinking these operations in the
  task is the simplest to avoid adding dependences.

  }];
}

def LowerDataflowTasks : Pass<"LowerDataflowTasks", "mlir::ModuleOp"> {
  let summary =
      "Outline the body of a DataflowTaskOp into a separate function which will serve as a task work function and lower the task graph to RT.";

  let description = [{
    This pass lowers a DataflowTaskGraph to the RT dialect, outlining
    DataflowTaskOp into separate work functions and introducing the
    necessary operations to communicate and synchronize execution via
    futures.
  }];
}



#endif
