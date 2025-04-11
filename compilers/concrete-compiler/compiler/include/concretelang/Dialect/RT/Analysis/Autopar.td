#ifndef CONCRETELANG_DIALECT_RT_ANALYSIS_AUTOPAR
#define CONCRETELANG_DIALECT_RT_ANALYSIS_AUTOPAR

include "mlir/Pass/PassBase.td"

def BuildDataflowTaskGraph : Pass<"BuildDataflowTaskGraph", "mlir::func::FuncOp"> {
  let summary =
      "Identify profitable dataflow tasks and build DataflowTaskGraph.";

  let description = [{
  This pass builds a dataflow graph out of a FHE program.

  In its current incarnation, it considers some heavier weight
  operations (e.g., FHELinalg Dot and Matmult or bootstraps) as
  candidates for being executed in a discrete task, and then
  sinks within the task the lighter weight operation that do not
  increase the graph cut (amount of dependences in or out).

  The output is a program partitioned in RT::DataflowTaskOp that
  expose task dependences as arguments and results of the
  DataflowTaskOp.

  Example:

```mlir
  func @main(%arg0: tensor<3x4x!FHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<3x2x!FHE.eint<2>> {
    %0 = "FHELinalg.matmul_eint_int"(%arg0, %arg1) : (tensor<3x4x!FHE.eint<2>>, tensor<4x2xi3>) -> tensor<3x2x!FHE.eint<2>>
    return %0 : tensor<3x2x!FHE.eint<2>>
  }
```

  Will result in generating a dataflow task for the Matmul operation:

```mlir
  func @main(%arg0: tensor<3x4x!FHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<3x2x!FHE.eint<2>> {
    %0 = "RT.dataflow_task"(%arg0, %arg1) ( {
      %1 = "FHELinalg.matmul_eint_int"(%arg0, %arg1) : (tensor<3x4x!FHE.eint<2>>, tensor<4x2xi3>) -> tensor<3x2x!FHE.eint<2>>
      "RT.dataflow_yield"(%1) : (tensor<3x2x!FHE.eint<2>>) -> ()
    }) : (tensor<3x4x!FHE.eint<2>>, tensor<4x2xi3>) -> tensor<3x2x!FHE.eint<2>>
    return %0 : tensor<3x2x!FHE.eint<2>>
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

def FinalizeTaskCreation : Pass<"FinalizeTaskCreation", "mlir::ModuleOp"> {
  let summary =
      "Finalize the CreateAsyncTaskOp ops.";

  let description = [{
      This pass adds the lower level information missing in
      CreateAsyncTaskOp, in particular the type sizes and if required
      passing the runtime context.
  }];
}

def StartStop : Pass<"dfr-add-start-stop", "mlir::ModuleOp"> {
  let summary =
      "Issue calls to start/stop the runtime system.";

  let description = [{
      This pass adds calls to _dfr_start and _dfr_stop which
      respectively initialize/start and pause the runtime system.  The
      start function further distributes the evaluation keys to
      compute nodes when required and the stop function clears the
      execution context.
  }];
}

def FixupBufferDeallocation : Pass<"FixupBufferDeallocation", "mlir::ModuleOp"> {
  let summary =
      "Prevent deallocation of buffers returned as futures by tasks.";

  let description = [{ This pass removes buffer deallocation calls on
      buffers being used for dataflow communication between
      tasks. These buffers cannot be deallocated directly without
      synchronization as they can be needed by asynchronous
      computation. Instead, these will be deallocated by the runtime
      when no longer needed.}]; }

#endif
