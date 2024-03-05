// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/TypeInference/IR/TypeInferenceOps.h"

namespace mlir {
namespace concretelang {
namespace TypeInference {} // namespace TypeInference
} // namespace concretelang
} // namespace mlir

#define GET_OP_CLASSES
#include "concretelang/Dialect/TypeInference/IR/TypeInferenceOps.cpp.inc"
