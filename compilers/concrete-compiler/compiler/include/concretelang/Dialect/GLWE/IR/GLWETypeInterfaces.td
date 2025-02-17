//===- GLWEInterfaces.td - High level GLWE interfaces -----*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
#ifndef GLWE_TYPE_INTERFACE
#define GLWE_TYPE_INTERFACE

include "mlir/IR/OpBase.td"

def GLWETypeInterface : TypeInterface<"GLWETypeInterface"> {
    let description = [{
        Interface to specify the minimal variances.
    }];

  let methods = [
    InterfaceMethod<"set the minimal variance expression for this type.",
                    "::mlir::Type", "withMinimalVariance",
                    (ins "NoiseDistribution":$noise_distribution, "int": $security_level)>,

    InterfaceMethod<"get variance expression for this type.",
                    "GLWEExpr", "getVariance",
                    (ins), [{
                      return $_type.getVariance().getExpr();
                    }]>,

    InterfaceMethod<"set variance expression for this type.",
                    "::mlir::Type", "withVariance",
                    (ins "GLWEExpr":$variance)>
  ];
}

#endif // SHAPE_INFERENCE_INTERFACE
