// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "concretelang-c/Dialect/HLFHELinalg.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.h"
#include "concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgOps.h"
#include "concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgTypes.h"

using namespace mlir::concretelang::HLFHELinalg;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HLFHELinalg, hlfhelinalg,
                                      HLFHELinalgDialect)
