#include "zamalang-c/Dialect/HLFHELinalg.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.h"
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgOps.h"
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgTypes.h"

using namespace mlir::zamalang::HLFHELinalg;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HLFHELinalg, hlfhelinalg,
                                      HLFHELinalgDialect)
