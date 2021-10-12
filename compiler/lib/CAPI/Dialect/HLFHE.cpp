#include "zamalang-c/Dialect/HLFHE.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"

using namespace mlir::zamalang::HLFHE;

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
