// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang-c/Dialect/FHE.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace mlir::concretelang::FHE;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FHE, fhe, FHEDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

bool fheTypeIsAnEncryptedIntegerType(MlirType type) {
  return unwrap(type).isa<EncryptedIntegerType>();
}

MlirType fheEncryptedIntegerTypeGetChecked(
    MlirContext ctx, unsigned width,
    mlir::function_ref<mlir::InFlightDiagnostic()> emitError) {
  return wrap(EncryptedIntegerType::getChecked(emitError, unwrap(ctx), width));
}
