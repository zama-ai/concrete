#ifndef ZAMALANG_C_DIALECT_HLFHELINALG_H
#define ZAMALANG_C_DIALECT_HLFHELINALG_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HLFHELinalg, hlfhelinalg);

#ifdef __cplusplus
}
#endif

#endif // ZAMALANG_C_DIALECT_HLFHELINALG_H
