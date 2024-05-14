// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/context.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include <assert.h>
#include <stdio.h>

namespace mlir {
namespace concretelang {

FFT::FFT(size_t polynomial_size)
    : fft(nullptr), polynomial_size(polynomial_size) {
  fft = (struct Fft *)aligned_alloc(CONCRETE_FFT_ALIGN, CONCRETE_FFT_SIZE);
  concrete_cpu_construct_concrete_fft(fft, polynomial_size);
}

FFT::FFT(FFT &&other) : fft(other.fft), polynomial_size(other.polynomial_size) {
  other.fft = nullptr;
}

FFT::~FFT() {
  if (fft != nullptr) {
    concrete_cpu_destroy_concrete_fft(fft);
    free(fft);
  }
}

RuntimeContext::RuntimeContext(ServerKeyset serverKeyset)
    : serverKeyset(serverKeyset) {

  // Initialize for each bootstrap key the fourier one
  for (size_t i = 0; i < serverKeyset.lweBootstrapKeys.size(); i++) {
    auto fdbsk = convert_to_fourier_domain(serverKeyset.lweBootstrapKeys[i]);
    // Store the fourier_bootstrap_key in the context
    fourier_bootstrap_keys.push_back(fdbsk.second);
    ffts.push_back(std::move(fdbsk.first));
  }

#ifdef CONCRETELANG_CUDA_SUPPORT
  assert(cudaGetDeviceCount(&num_devices) == cudaSuccess);
  bsk_gpu.resize(num_devices);
  ksk_gpu.resize(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    bsk_gpu[i].resize(serverKeyset.lweBootstrapKeys.size(), nullptr);
    ksk_gpu[i].resize(serverKeyset.lweKeyswitchKeys.size(), nullptr);
    bsk_gpu_mutex.push_back(std::make_unique<std::mutex>());
    ksk_gpu_mutex.push_back(std::make_unique<std::mutex>());
  }
#endif
}

std::pair<FFT, std::shared_ptr<std::vector<std::complex<double>>>>
RuntimeContext::convert_to_fourier_domain(LweBootstrapKey &bsk) {
  auto info = bsk.getInfo().asReader();

  size_t decomposition_level_count = info.getParams().getLevelCount();
  size_t decomposition_base_log = info.getParams().getBaseLog();
  size_t glwe_dimension = info.getParams().getGlweDimension();
  size_t polynomial_size = info.getParams().getPolynomialSize();
  size_t input_lwe_dimension = info.getParams().getInputLweDimension();

  // Create the FFT
  FFT fft(polynomial_size);

  // Allocate scratch for key conversion
  size_t scratch_size;
  size_t scratch_align;
  concrete_cpu_bootstrap_key_convert_u64_to_fourier_scratch(
      &scratch_size, &scratch_align, fft.fft);
  auto scratch = (uint8_t *)aligned_alloc(scratch_align, scratch_size);

  // Allocate the fourier_bootstrap_key
  auto &bsk_buffer = bsk.getBuffer();
  auto fourier_data = std::make_shared<std::vector<std::complex<double>>>();
  fourier_data->resize(bsk_buffer.size() / 2);
  auto bsk_data = bsk_buffer.data();

  // Convert bootstrap_key to the fourier domain
  concrete_cpu_bootstrap_key_convert_u64_to_fourier(
      bsk_data, fourier_data->data(), decomposition_level_count,
      decomposition_base_log, glwe_dimension, polynomial_size,
      input_lwe_dimension, fft.fft, scratch, scratch_size);
  free(scratch);

  return std::pair<FFT, std::shared_ptr<std::vector<std::complex<double>>>>(
      std::move(fft), fourier_data);
}
} // namespace concretelang
} // namespace mlir

#ifdef CONCRETELANG_DATAFLOW_EXECUTION_ENABLED
#include "concretelang/Runtime/key_manager.hpp"

// Register the HPX actions for retrieving the evaluation keys from
// the master node (must be in global namespace)
HPX_PLAIN_ACTION(mlir::concretelang::dfr::getKsk, _dfr_get_ksk_action)
HPX_PLAIN_ACTION(mlir::concretelang::dfr::getBsk, _dfr_get_bsk_action)
HPX_PLAIN_ACTION(mlir::concretelang::dfr::getPKsk, _dfr_get_pksk_action)

namespace mlir {
namespace concretelang {
const uint64_t *DistributedRuntimeContext::keyswitch_key_buffer(size_t keyId) {
  if (dfr::_dfr_is_root_node())
    return RuntimeContext::keyswitch_key_buffer(keyId);

  std::lock_guard<std::mutex> guard(cm_guard);
  if (ksks.find(keyId) == ksks.end()) {
    _dfr_get_ksk_action getKskAction;
    dfr::KeyWrapper<LweKeyswitchKey> kskw =
        getKskAction(hpx::find_root_locality(), keyId);
    ksks.insert(std::pair<size_t, LweKeyswitchKey>(keyId, kskw.keys[0]));
  }
  auto it = ksks.find(keyId);
  assert(it != ksks.end());
  return it->second.getBuffer().data();
}

void DistributedRuntimeContext::getBSKonNode(size_t keyId) {
  assert(fbks.find(keyId) == fbks.end());
  assert(dffts.find(keyId) == dffts.end());
  _dfr_get_bsk_action getBskAction;
  dfr::KeyWrapper<LweBootstrapKey> bskw =
      getBskAction(hpx::find_root_locality(), keyId);

  auto fdbsk = convert_to_fourier_domain(bskw.keys[0]);
  fbks.insert(
      std::pair<size_t, std::shared_ptr<std::vector<std::complex<double>>>>(
          keyId, fdbsk.second));
  dffts.insert(std::pair<size_t, FFT>(keyId, std::move(fdbsk.first)));
}

const std::complex<double> *
DistributedRuntimeContext::fourier_bootstrap_key_buffer(size_t keyId) {
  if (dfr::_dfr_is_root_node())
    return RuntimeContext::fourier_bootstrap_key_buffer(keyId);

  std::lock_guard<std::mutex> guard(cm_guard);
  if (fbks.find(keyId) == fbks.end())
    getBSKonNode(keyId);
  auto it = fbks.find(keyId);
  assert(it != fbks.end());
  return it->second->data();
}

const uint64_t *
DistributedRuntimeContext::fp_keyswitch_key_buffer(size_t keyId) {
  if (dfr::_dfr_is_root_node())
    return RuntimeContext::fp_keyswitch_key_buffer(keyId);

  std::lock_guard<std::mutex> guard(cm_guard);
  if (ksks.find(keyId) == ksks.end()) {
    _dfr_get_pksk_action getPKskAction;
    dfr::KeyWrapper<PackingKeyswitchKey> pkskw =
        getPKskAction(hpx::find_root_locality(), keyId);
    pksks.insert(std::pair<size_t, PackingKeyswitchKey>(keyId, pkskw.keys[0]));
  }
  auto it = pksks.find(keyId);
  assert(it != pksks.end());
  return it->second.getRawPtr();
}

const struct Fft *DistributedRuntimeContext::fft(size_t keyId) {
  if (dfr::_dfr_is_root_node())
    return RuntimeContext::fft(keyId);

  std::lock_guard<std::mutex> guard(cm_guard);
  if (dffts.find(keyId) == dffts.end())
    getBSKonNode(keyId);
  auto it = dffts.find(keyId);
  assert(it != dffts.end());
  return it->second.fft;
}

} // namespace concretelang
} // namespace mlir
#endif
