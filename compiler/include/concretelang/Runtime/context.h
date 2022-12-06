// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_CONTEXT_H
#define CONCRETELANG_RUNTIME_CONTEXT_H

#include <assert.h>
#include <map>
#include <mutex>
#include <pthread.h>

#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/Runtime/seeder.h"

#include "concrete-core-ffi.h"
#include "concretelang/Common/Error.h"

#ifdef CONCRETELANG_CUDA_SUPPORT
#include "bootstrap.h"
#include "device.h"
#include "keyswitch.h"
#endif

namespace mlir {
namespace concretelang {

typedef struct RuntimeContext {

  RuntimeContext() {
    CAPI_ASSERT_ERROR(new_default_engine(best_seeder, &default_engine));
#ifdef CONCRETELANG_CUDA_SUPPORT
    bsk_gpu = nullptr;
    ksk_gpu = nullptr;
#endif
  }

  /// Ensure that the engines map is not copied
  RuntimeContext(const RuntimeContext &ctx){};

  ~RuntimeContext() {
    CAPI_ASSERT_ERROR(destroy_default_engine(default_engine));
    for (const auto &key : fft_engines) {
      CAPI_ASSERT_ERROR(destroy_fft_engine(key.second));
    }
    if (fbsk != nullptr) {
      CAPI_ASSERT_ERROR(destroy_fft_fourier_lwe_bootstrap_key_u64(fbsk));
    }
#ifdef CONCRETELANG_CUDA_SUPPORT
    if (bsk_gpu != nullptr) {
      cuda_drop(bsk_gpu, 0);
    }
    if (ksk_gpu != nullptr) {
      cuda_drop(ksk_gpu, 0);
    }
#endif
  }

  FftEngine *get_fft_engine() {
    pthread_t threadId = pthread_self();
    std::lock_guard<std::mutex> guard(engines_map_guard);
    auto engineIt = fft_engines.find(threadId);
    if (engineIt == fft_engines.end()) {
      FftEngine *fft_engine = nullptr;

      CAPI_ASSERT_ERROR(new_fft_engine(&fft_engine));

      engineIt =
          fft_engines
              .insert(std::pair<pthread_t, FftEngine *>(threadId, fft_engine))
              .first;
    }
    assert(engineIt->second && "No engine available in context");
    return engineIt->second;
  }

  DefaultEngine *get_default_engine() { return default_engine; }

  FftFourierLweBootstrapKey64 *get_fft_fourier_bsk() {

    if (fbsk != nullptr) {
      return fbsk;
    }

    const std::lock_guard<std::mutex> guard(fbskMutex);
    if (fbsk == nullptr) {
      CAPI_ASSERT_ERROR(
          fft_engine_convert_lwe_bootstrap_key_to_fft_fourier_lwe_bootstrap_key_u64(
              get_fft_engine(), evaluationKeys.getBsk(), &fbsk));
    }
    return fbsk;
  }

#ifdef CONCRETELANG_CUDA_SUPPORT
  void *get_bsk_gpu(uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
                    uint32_t glwe_dim, uint32_t gpu_idx, void *stream) {

    if (bsk_gpu != nullptr) {
      return bsk_gpu;
    }
    const std::lock_guard<std::mutex> guard(bsk_gpu_mutex);

    if (bsk_gpu != nullptr) {
      return bsk_gpu;
    }
    LweBootstrapKey64 *bsk = get_bsk();
    size_t bsk_buffer_len =
        input_lwe_dim * (glwe_dim + 1) * (glwe_dim + 1) * poly_size * level;
    size_t bsk_buffer_size = bsk_buffer_len * sizeof(uint64_t);
    uint64_t *bsk_buffer =
        (uint64_t *)aligned_alloc(U64_ALIGNMENT, bsk_buffer_size);
    size_t bsk_gpu_buffer_size = bsk_buffer_len * sizeof(double);
    bsk_gpu = cuda_malloc(bsk_gpu_buffer_size, gpu_idx);
    CAPI_ASSERT_ERROR(
        default_engine_discard_convert_lwe_bootstrap_key_to_lwe_bootstrap_key_mut_view_u64_raw_ptr_buffers(
            default_engine, bsk, bsk_buffer));
    cuda_initialize_twiddles(poly_size, gpu_idx);
    cuda_convert_lwe_bootstrap_key_64(bsk_gpu, bsk_buffer, stream, gpu_idx,
                                      input_lwe_dim, glwe_dim, level,
                                      poly_size);
    // This is currently not 100% async as we have to free CPU memory after
    // conversion
    cuda_synchronize_device(gpu_idx);
    free(bsk_buffer);
    return bsk_gpu;
  }

  void *get_ksk_gpu(uint32_t level, uint32_t input_lwe_dim,
                    uint32_t output_lwe_dim, uint32_t gpu_idx, void *stream) {

    if (ksk_gpu != nullptr) {
      return ksk_gpu;
    }

    const std::lock_guard<std::mutex> guard(ksk_gpu_mutex);
    if (ksk_gpu != nullptr) {
      return ksk_gpu;
    }
    LweKeyswitchKey64 *ksk = get_ksk();
    size_t ksk_buffer_len = input_lwe_dim * (output_lwe_dim + 1) * level;
    size_t ksk_buffer_size = sizeof(uint64_t) * ksk_buffer_len;
    uint64_t *ksk_buffer =
        (uint64_t *)aligned_alloc(U64_ALIGNMENT, ksk_buffer_size);
    void *ksk_gpu = cuda_malloc(ksk_buffer_size, gpu_idx);
    CAPI_ASSERT_ERROR(
        default_engine_discard_convert_lwe_keyswitch_key_to_lwe_keyswitch_key_mut_view_u64_raw_ptr_buffers(
            default_engine, ksk, ksk_buffer));
    cuda_memcpy_async_to_gpu(ksk_gpu, ksk_buffer, ksk_buffer_size, stream,
                             gpu_idx);
    // This is currently not 100% async as we have to free CPU memory after
    // conversion
    cuda_synchronize_device(gpu_idx);
    free(ksk_buffer);
    return ksk_gpu;
  }
#endif

  LweBootstrapKey64 *get_bsk() { return evaluationKeys.getBsk(); }

  LweKeyswitchKey64 *get_ksk() { return evaluationKeys.getKsk(); }

  LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys64 *get_fpksk() {
    return evaluationKeys.getFpksk();
  }

  RuntimeContext &operator=(const RuntimeContext &rhs) {
    this->evaluationKeys = rhs.evaluationKeys;
    return *this;
  }

  ::concretelang::clientlib::EvaluationKeys evaluationKeys;

private:
  std::mutex fbskMutex;
  FftFourierLweBootstrapKey64 *fbsk = nullptr;
  DefaultEngine *default_engine;
  std::map<pthread_t, FftEngine *> fft_engines;
  std::mutex engines_map_guard;

#ifdef CONCRETELANG_CUDA_SUPPORT
  std::mutex bsk_gpu_mutex;
  void *bsk_gpu;
  std::mutex ksk_gpu_mutex;
  void *ksk_gpu;
#endif

} RuntimeContext;

} // namespace concretelang
} // namespace mlir

extern "C" {
LweKeyswitchKey64 *
get_keyswitch_key_u64(mlir::concretelang::RuntimeContext *context);

FftFourierLweBootstrapKey64 *
get_fft_fourier_bootstrap_key_u64(mlir::concretelang::RuntimeContext *context);

LweBootstrapKey64 *
get_bootstrap_key_u64(mlir::concretelang::RuntimeContext *context);

DefaultEngine *get_engine(mlir::concretelang::RuntimeContext *context);

FftEngine *get_fft_engine(mlir::concretelang::RuntimeContext *context);
}
#endif
