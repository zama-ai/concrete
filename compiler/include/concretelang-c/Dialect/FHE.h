// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_C_DIALECT_FHE_H
#define CONCRETELANG_C_DIALECT_FHE_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FHE, fhe);

/// Creates an encrypted integer type of `width` bits
MLIR_CAPI_EXPORTED MlirType fheEncryptedIntegerTypeGetChecked(
    MlirContext context, unsigned width,
    mlir::function_ref<mlir::InFlightDiagnostic()> emitError);

/// If the type is an EncryptedInteger
MLIR_CAPI_EXPORTED bool fheTypeIsAnEncryptedIntegerType(MlirType);

#ifdef __cplusplus
}
#endif

#endif // CONCRETELANG_C_DIALECT_FHE_H
