// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "concretelang-c/Dialect/FHELinalg.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace mlir::concretelang::FHELinalg;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FHELinalg, fhelinalg, FHELinalgDialect)
