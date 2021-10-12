#ifndef ZAMALANG_C_DIALECT_HLFHE_H
#define ZAMALANG_C_DIALECT_HLFHE_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HLFHE, hlfhe);

/// Creates an encrypted integer type of `width` bits
MLIR_CAPI_EXPORTED MlirType hlfheEncryptedIntegerTypeGetChecked(
    MlirContext context, unsigned width,
    mlir::function_ref<mlir::InFlightDiagnostic()> emitError);

/// If the type is an EncryptedInteger
MLIR_CAPI_EXPORTED bool hlfheTypeIsAnEncryptedIntegerType(MlirType);

#ifdef __cplusplus
}
#endif

#endif // ZAMALANG_C_DIALECT_HLFHE_H
