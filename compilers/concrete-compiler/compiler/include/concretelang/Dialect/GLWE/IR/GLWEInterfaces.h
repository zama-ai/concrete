
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWE_INTERFACE_H_
#define CONCRETELANG_DIALECT_GLWE_IR_GLWE_INTERFACE_H_

#include "mlir/IR/OpDefinition.h"

#include "concretelang/Common/Error.h"
#include "concretelang/Dialect/GLWE/IR/GLWEEnums.h.inc"
#include "concretelang/Dialect/GLWE/IR/GLWEExpr.h"

namespace mlir {
namespace concretelang {
namespace GLWE {
/// Include the auto-generated declarations.
#include "concretelang/Dialect/GLWE/IR/GLWEInterfaces.h.inc"
#include "concretelang/Dialect/GLWE/IR/GLWETypeInterfaces.h.inc"
} // namespace GLWE
} // namespace concretelang
} // namespace mlir

#endif
