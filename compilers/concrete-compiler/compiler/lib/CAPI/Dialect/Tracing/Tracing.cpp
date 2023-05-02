// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang-c/Dialect/Tracing.h"
#include "concretelang/Dialect/Tracing/IR/TracingDialect.h"
#include "concretelang/Dialect/Tracing/IR/TracingOps.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace mlir::concretelang::Tracing;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Tracing, tracing, TracingDialect)
