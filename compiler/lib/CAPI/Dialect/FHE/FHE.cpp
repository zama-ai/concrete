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
#include "mlir/IR/StorageUniquerSupport.h"

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

MlirTypeOrError fheEncryptedIntegerTypeGetChecked(MlirContext ctx,
                                                  unsigned width) {
  MlirTypeOrError type = {{NULL}, false};
  auto catchError = [&]() -> mlir::InFlightDiagnostic {
    type.isError = true;
    mlir::DiagnosticEngine &engine = unwrap(ctx)->getDiagEngine();
    // The goal here is to make getChecked working, but we don't want the CAPI
    // to stop execution due to an error, and leave the error handling logic to
    // the user of the CAPI
    return engine.emit(mlir::UnknownLoc::get(unwrap(ctx)),
                       mlir::DiagnosticSeverity::Warning);
  };
  EncryptedIntegerType eint =
      EncryptedIntegerType::getChecked(catchError, unwrap(ctx), width);
  if (type.isError) {
    return type;
  }
  type.type = wrap(eint);
  return type;
}
