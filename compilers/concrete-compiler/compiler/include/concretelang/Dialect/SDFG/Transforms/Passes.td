#ifndef CONCRETELANG_SDFG_PASSES
#define CONCRETELANG_SDFG_PASSES

include "mlir/Pass/PassBase.td"

def SDFGBufferOwnership : Pass<"SDFGBufferOwnership", "mlir::ModuleOp"> {
  let summary =
      "Take ownership of data passed to SDFG operators.";

  let description = [{ As data used in SDFG operators, in particular
  PUT operations, can be used asynchronously, deallocation must be
  handled by the runtime if we take ownership.  This pass removes
  explicit deallocation calls where no other uses of the data exist
  and makes copies otherwise, letting the runtime handle
  deallocation when appropriate.}]; }

#endif
