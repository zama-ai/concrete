#ifndef CONCRETELANG_ONE_SHOT_BUFFERIZE_DPS_WRAPPER_PASS
#define CONCRETELANG_ONE_SHOT_BUFFERIZE_DPS_WRAPPER_PASS

include "mlir/Pass/PassBase.td"

def OneShotBufferizeDPSWrapper
    : Pass<"one-shot-bufferize-dps-wrapper", "::mlir::ModuleOp"> {
  let summary =
      "Converts functions to destination-passing and generates a wrapper "
      "function allocating and returning memrefs for return values";

  let description = [{
      The one-shot bufferizer converts all functions returning tensor values
      to functions using destination-passing style with one output memref for
      each output value. In order to support external callers not using
      destination-passing style and expecting memrefs to be returned, this
      pass generates a wrapper function that allocates the corresponding
      memref for each output tensor of the original function, invokes the
      function using destination-passing style and returns the allocated
      memrefs to the caller.

      Example:

      ```
      func @main(...) -> tensor<3x2049xi64> {
        ...
      }
      ```

      becomes:

      ```
      func private @main(...) -> memref<3x2049xi64> {
        %0 = memref.alloc() : memref<3x2049xi64>
        call @__dps_main(..., %0) : (..., memref<3x2049xi64>) -> ()
        return %0 : memref<3x2049xi64>
      }
      
      func @__dps_main(..., tensor<3x2049xi64>) {
        ...
      }
      ```
  }];

  let constructor =
      "mlir::concretelang::createOneShotBufferizeDPSWrapperPass()";

  let options = [];

  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect", "mlir::memref::MemRefDialect"
  ];
}

#endif
