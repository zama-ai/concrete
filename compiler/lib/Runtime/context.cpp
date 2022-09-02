// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/context.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Runtime/seeder.h"
#include <assert.h>
#include <stdio.h>

LweKeyswitchKey64 *
get_keyswitch_key_u64(mlir::concretelang::RuntimeContext *context) {
  return context->get_ksk();
}

LweBootstrapKey64 *
get_bootstrap_key_u64(mlir::concretelang::RuntimeContext *context) {
  return context->get_bsk();
}

FftwFourierLweBootstrapKey64 *get_fftw_fourier_bootstrap_key_u64(
    mlir::concretelang::RuntimeContext *context) {
  return context->get_fftw_fourier_bsk();
}

DefaultEngine *get_engine(mlir::concretelang::RuntimeContext *context) {
  return context->get_default_engine();
}

FftwEngine *get_fftw_engine(mlir::concretelang::RuntimeContext *context) {
  return context->get_fftw_engine();
}
