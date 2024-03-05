// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/TypeInference/IR/TypeInferenceDialect.h"
#include "concretelang/Dialect/TypeInference/IR/TypeInferenceOps.h"

#include "concretelang/Dialect/TypeInference/IR/TypeInferenceOpsDialect.cpp.inc"

using namespace mlir::concretelang::TypeInference;

void TypeInferenceDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/TypeInference/IR/TypeInferenceOps.cpp.inc"
      >();
}
