// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "concretelang-c/Dialect/HLFHE.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "concretelang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "concretelang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "concretelang/Dialect/HLFHE/IR/HLFHETypes.h"

using namespace mlir::concretelang::HLFHE;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HLFHE, hlfhe, HLFHEDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

bool hlfheTypeIsAnEncryptedIntegerType(MlirType type) {
  return unwrap(type).isa<EncryptedIntegerType>();
}

MlirType hlfheEncryptedIntegerTypeGetChecked(
    MlirContext ctx, unsigned width,
    mlir::function_ref<mlir::InFlightDiagnostic()> emitError) {
  return wrap(EncryptedIntegerType::getChecked(emitError, unwrap(ctx), width));
}
