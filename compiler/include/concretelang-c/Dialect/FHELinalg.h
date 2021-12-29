// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef CONCRETELANG_C_DIALECT_FHELINALG_H
#define CONCRETELANG_C_DIALECT_FHELINALG_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FHELinalg, fhelinalg);

#ifdef __cplusplus
}
#endif

#endif // CONCRETELANG_C_DIALECT_FHELINALG_H
