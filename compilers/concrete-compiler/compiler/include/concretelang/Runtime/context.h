// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_CONTEXT_H
#define CONCRETELANG_RUNTIME_CONTEXT_H

#include "concrete-cpu.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include <assert.h>
#include <complex>
#include <map>
#include <mutex>
#include <pthread.h>
#include <vector>

using ::concretelang::keysets::ServerKeyset;

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
  FFT(const FFT &other) = delete;
  FFT(FFT &&other);
  ~FFT();

  struct Fft *fft;
  size_t polynomial_size;
} FFT;

typedef struct RuntimeContext {

  RuntimeContext() = delete;
  RuntimeContext(ServerKeyset serverKeyset);
  virtual ~RuntimeContext() {
#ifdef CONCRETELANG_CUDA_SUPPORT
    for (int i = 0; i < num_devices; ++i) {
      for (auto k : bsk_gpu[i])
        if (k != nullptr)
          cuda_drop(k, i);
      for (auto k : ksk_gpu[i])
        if (k != nullptr)
          cuda_drop(k, i);
    }
#endif
  };

  virtual const uint64_t *keyswitch_key_buffer(size_t keyId) {
    return serverKeyset.lweKeyswitchKeys[keyId].getBuffer().data();
  }

  virtual const std::complex<double> *
  fourier_bootstrap_key_buffer(size_t keyId) {
    return fourier_bootstrap_keys[keyId]->data();
  }

  virtual const uint64_t *fp_keyswitch_key_buffer(size_t keyId) {
    return serverKeyset.packingKeyswitchKeys[keyId].getRawPtr();
  }

  virtual const struct Fft *fft(size_t keyId) { return ffts[keyId].fft; }

  const ServerKeyset getKeys() const { return serverKeyset; }

protected:
  ServerKeyset serverKeyset;
  std::vector<std::shared_ptr<std::vector<std::complex<double>>>>
      fourier_bootstrap_keys;
  std::vector<FFT> ffts;
  std::pair<FFT, std::shared_ptr<std::vector<std::complex<double>>>>
  convert_to_fourier_domain(LweBootstrapKey &bsk);

#ifdef CONCRETELANG_CUDA_SUPPORT
public:
  void *get_bsk_gpu(uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
                    uint32_t glwe_dim, uint32_t gpu_idx, void *stream,
                    uint32_t bsk_idx) {

    if (bsk_gpu[gpu_idx][bsk_idx] != nullptr) {
      return bsk_gpu[gpu_idx][bsk_idx];
    }
    const std::lock_guard<std::mutex> guard(*bsk_gpu_mutex[gpu_idx]);

    if (bsk_gpu[gpu_idx][bsk_idx] != nullptr) {
      return bsk_gpu[gpu_idx][bsk_idx];
    }

    auto bsk = serverKeyset.lweBootstrapKeys[bsk_idx];

    size_t bsk_buffer_len = bsk.getBuffer().size();
    size_t bsk_gpu_buffer_size = bsk_buffer_len * sizeof(double);

    void *bsk_gpu_tmp =
        cuda_malloc_async(bsk_gpu_buffer_size, (cudaStream_t *)stream, gpu_idx);
    cuda_convert_lwe_bootstrap_key_64(
        bsk_gpu_tmp, const_cast<uint64_t *>(bsk.getBuffer().data()),
        (cudaStream_t *)stream, gpu_idx, input_lwe_dim, glwe_dim, level,
        poly_size);
    // Synchronization here is not optional as it works with mutex to
    // prevent other GPU streams from reading partially copied keys.
    cudaStreamSynchronize(*(cudaStream_t *)stream);
    bsk_gpu[gpu_idx][bsk_idx] = bsk_gpu_tmp;
    return bsk_gpu[gpu_idx][bsk_idx];
  }

  void *get_ksk_gpu(uint32_t level, uint32_t input_lwe_dim,
                    uint32_t output_lwe_dim, uint32_t gpu_idx, void *stream,
                    uint32_t ksk_idx) {

    if (ksk_gpu[gpu_idx][ksk_idx] != nullptr) {
      return ksk_gpu[gpu_idx][ksk_idx];
    }

    const std::lock_guard<std::mutex> guard(*ksk_gpu_mutex[gpu_idx]);
    if (ksk_gpu[gpu_idx][ksk_idx] != nullptr) {
      return ksk_gpu[gpu_idx][ksk_idx];
    }

    auto ksk = serverKeyset.lweKeyswitchKeys[ksk_idx];

    size_t ksk_buffer_size = sizeof(uint64_t) * ksk.getBuffer().size();

    void *ksk_gpu_tmp =
        cuda_malloc_async(ksk_buffer_size, (cudaStream_t *)stream, gpu_idx);

    cuda_memcpy_async_to_gpu(ksk_gpu_tmp,
                             const_cast<uint64_t *>(ksk.getBuffer().data()),
                             ksk_buffer_size, (cudaStream_t *)stream, gpu_idx);
    // Synchronization here is not optional as it works with mutex to
    // prevent other GPU streams from reading partially copied keys.
    cudaStreamSynchronize(*(cudaStream_t *)stream);
    ksk_gpu[gpu_idx][ksk_idx] = ksk_gpu_tmp;
    return ksk_gpu[gpu_idx][ksk_idx];
  }

private:
  std::vector<std::unique_ptr<std::mutex>> bsk_gpu_mutex;
  std::vector<std::vector<void *>> bsk_gpu;
  std::vector<std::unique_ptr<std::mutex>> ksk_gpu_mutex;
  std::vector<std::vector<void *>> ksk_gpu;
  int num_devices;
#endif
} RuntimeContext;

struct DistributedRuntimeContext : public RuntimeContext {

  using RuntimeContext::RuntimeContext;
  const uint64_t *keyswitch_key_buffer(size_t keyId) override;
  const std::complex<double> *
  fourier_bootstrap_key_buffer(size_t keyId) override;
  const uint64_t *fp_keyswitch_key_buffer(size_t keyId) override;
  const struct Fft *fft(size_t keyId) override;

private:
  void getBSKonNode(size_t keyId);
  std::mutex cm_guard;
  std::map<size_t, LweKeyswitchKey> ksks;
  std::map<size_t, std::shared_ptr<std::vector<std::complex<double>>>> fbks;
  std::map<size_t, FFT> dffts;
  std::map<size_t, PackingKeyswitchKey> pksks;
};

} // namespace concretelang
} // namespace mlir

#endif
