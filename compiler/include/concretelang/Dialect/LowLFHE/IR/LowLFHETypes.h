// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef CONCRETELANG_DIALECT_LowLFHE_IR_LowLFHETYPES_H
#define CONCRETELANG_DIALECT_LowLFHE_IR_LowLFHETYPES_H

#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/LowLFHE/IR/LowLFHEOpsTypes.h.inc"

#endif
