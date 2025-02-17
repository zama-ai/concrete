//===- GLWEInterfaces.td - High level GLWE interfaces -----*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
#ifndef GLWE_OP_INTERFACE
#define GLWE_OP_INTERFACE

include "mlir/IR/OpBase.td"

def GLWEOpInterface : OpInterface<"GLWEOpInterface"> {
    let description = [{
        Interface to specify variances and cost methods that are used
        in the GLWE optimization pipeline.
    }];

  let methods = [
    InterfaceMethod<"Infer and return the default variance expression for this op.",
                    "GLWEExpr", "defaultVariance">,
    InterfaceMethod<"Resolve and return the ouput variance expression for this op.",
                    "::concretelang::error::Result<GLWEExpr>", "resolveOutputVariance">
  ];
}

#endif // SHAPE_INFERENCE_INTERFACE
