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
#include <vector>

#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/Common/Error.h"

#include "concrete-cpu.h"

#ifdef CONCRETELANG_CUDA_SUPPORT
#include "bootstrap.h"
#include "device.h"
#include "keyswitch.h"
#endif

namespace mlir {
namespace concretelang {

typedef struct FFT {
  FFT() = delete;
  FFT(size_t polynomial_size);
  FFT(FFT &other) = delete;
  FFT(FFT &&other);
  ~FFT();

  struct Fft *fft;
  size_t polynomial_size;
} FFT;

typedef struct RuntimeContext {

  RuntimeContext() = delete;
  RuntimeContext(::concretelang::clientlib::EvaluationKeys evaluationKeys);
  ~RuntimeContext() {
#ifdef CONCRETELANG_CUDA_SUPPORT
    for (int i = 0; i < num_devices; ++i) {
      if (bsk_gpu[i] != nullptr)
        cuda_drop(bsk_gpu[i], i);
      if (ksk_gpu[i] != nullptr)
        cuda_drop(ksk_gpu[i], i);
    }
#endif
  };

  const uint64_t *keyswitch_key_buffer(size_t keyId) {
    return evaluationKeys.getKeyswitchKey(keyId).buffer();
  }

  const double *fourier_bootstrap_key_buffer(size_t keyId) {
    return fourier_bootstrap_keys[keyId]->data();
  }

  const uint64_t *fp_keyswitch_key_buffer(size_t keyId) {
    return evaluationKeys.getPackingKeyswitchKey(keyId).buffer();
  }

  const struct Fft *fft(size_t keyId) { return ffts[keyId].fft; }

  const ::concretelang::clientlib::EvaluationKeys getKeys() const {
    return evaluationKeys;
  }

private:
  ::concretelang::clientlib::EvaluationKeys evaluationKeys;
  std::vector<std::shared_ptr<std::vector<double>>> fourier_bootstrap_keys;
  std::vector<FFT> ffts;

#ifdef CONCRETELANG_CUDA_SUPPORT
public:
  void *get_bsk_gpu(uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
                    uint32_t glwe_dim, uint32_t gpu_idx, void *stream) {

    if (bsk_gpu[gpu_idx] != nullptr) {
      return bsk_gpu[gpu_idx];
    }
    const std::lock_guard<std::mutex> guard(*bsk_gpu_mutex[gpu_idx]);

    if (bsk_gpu[gpu_idx] != nullptr) {
      return bsk_gpu[gpu_idx];
    }

    auto bsk = evaluationKeys.getBootstrapKey(0);

    size_t bsk_buffer_len = bsk.size();
    size_t bsk_gpu_buffer_size = bsk_buffer_len * sizeof(double);

    void *bsk_gpu_tmp =
        cuda_malloc_async(bsk_gpu_buffer_size, (cudaStream_t *)stream, gpu_idx);
    cuda_convert_lwe_bootstrap_key_64(
        bsk_gpu_tmp, const_cast<uint64_t *>(bsk.buffer()),
        (cudaStream_t *)stream, gpu_idx, input_lwe_dim, glwe_dim, level,
        poly_size);
    // Synchronization here is not optional as it works with mutex to
    // prevent other GPU streams from reading partially copied keys.
    cudaStreamSynchronize(*(cudaStream_t *)stream);
    bsk_gpu[gpu_idx] = bsk_gpu_tmp;
    return bsk_gpu[gpu_idx];
  }

  void *get_ksk_gpu(uint32_t level, uint32_t input_lwe_dim,
                    uint32_t output_lwe_dim, uint32_t gpu_idx, void *stream) {

    if (ksk_gpu[gpu_idx] != nullptr) {
      return ksk_gpu[gpu_idx];
    }

    const std::lock_guard<std::mutex> guard(*ksk_gpu_mutex[gpu_idx]);
    if (ksk_gpu[gpu_idx] != nullptr) {
      return ksk_gpu[gpu_idx];
    }
    auto ksk = evaluationKeys.getKeyswitchKey(0);

    size_t ksk_buffer_size = sizeof(uint64_t) * ksk.size();

    void *ksk_gpu_tmp =
        cuda_malloc_async(ksk_buffer_size, (cudaStream_t *)stream, gpu_idx);

    cuda_memcpy_async_to_gpu(ksk_gpu_tmp, const_cast<uint64_t *>(ksk.buffer()),
                             ksk_buffer_size, (cudaStream_t *)stream, gpu_idx);
    // Synchronization here is not optional as it works with mutex to
    // prevent other GPU streams from reading partially copied keys.
    cudaStreamSynchronize(*(cudaStream_t *)stream);
    ksk_gpu[gpu_idx] = ksk_gpu_tmp;
    return ksk_gpu[gpu_idx];
  }

private:
  std::vector<std::unique_ptr<std::mutex>> bsk_gpu_mutex;
  std::vector<void *> bsk_gpu;
  std::vector<std::unique_ptr<std::mutex>> ksk_gpu_mutex;
  std::vector<void *> ksk_gpu;
  int num_devices;
#endif
} RuntimeContext;

} // namespace concretelang
} // namespace mlir

#endif
