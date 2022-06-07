// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>
#include <stdlib.h>

extern "C" {
#include "concrete-ffi.h"
}

#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/ClientLib/Serializers.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

// TODO: optimize the move
PublicArguments::PublicArguments(const ClientParameters &clientParameters,
                                 std::vector<void *> &&preparedArgs_,
                                 std::vector<TensorData> &&ciphertextBuffers_)
    : clientParameters(clientParameters) {
  preparedArgs = std::move(preparedArgs_);
  ciphertextBuffers = std::move(ciphertextBuffers_);
}

PublicArguments::~PublicArguments() {}

outcome::checked<void, StringError>
PublicArguments::serialize(std::ostream &ostream) {
  if (incorrectMode(ostream)) {
    return StringError(
        "PublicArguments::serialize: ostream should be in binary mode");
  }
  size_t iPreparedArgs = 0;
  int iGate = -1;
  for (auto gate : clientParameters.inputs) {
    iGate++;
    size_t rank = gate.shape.dimensions.size();
    if (!gate.encryption.hasValue()) {
      return StringError("PublicArguments::serialize: Clear arguments "
                         "are not yet supported. Argument ")
             << iGate;
    }
    /*auto allocated = */ preparedArgs[iPreparedArgs++];
    auto aligned = (encrypted_scalars_t)preparedArgs[iPreparedArgs++];
    assert(aligned != nullptr);
    auto offset = (size_t)preparedArgs[iPreparedArgs++];
    std::vector<int64_t> sizes; // includes lweSize as last dim
    sizes.resize(rank + 1);
    for (auto dim = 0u; dim < sizes.size(); dim++) {
      // sizes are part of the client parameters signature
      // it's static now but some day it could be dynamic so we serialize
      // them.
      sizes[dim] = (size_t)preparedArgs[iPreparedArgs++];
    }
    std::vector<size_t> strides(rank + 1);
    /* strides should be zero here and are not serialized */
    for (auto dim = 0u; dim < strides.size(); dim++) {
      strides[dim] = (size_t)preparedArgs[iPreparedArgs++];
    }
    // TODO: STRIDES
    auto values = aligned + offset;
    serializeTensorData(sizes, values, ostream);
  }
  return outcome::success();
}

outcome::checked<void, StringError>
PublicArguments::unserializeArgs(std::istream &istream) {
  int iGate = -1;
  for (auto gate : clientParameters.inputs) {
    iGate++;
    if (!gate.encryption.hasValue()) {
      return StringError("Clear values are not handled");
    }
    auto lweSize = clientParameters.lweSecretKeyParam(gate).value().lweSize();
    std::vector<int64_t> sizes = gate.shape.dimensions;
    sizes.push_back(lweSize);
    ciphertextBuffers.push_back(unserializeTensorData(sizes, istream));
    auto &values_and_sizes = ciphertextBuffers.back();
    if (istream.fail()) {
      return StringError(
                 "PublicArguments::unserializeArgs: Failed to read argument ")
             << iGate;
    }
    preparedArgs.push_back(/*allocated*/ nullptr);
    preparedArgs.push_back((void *)values_and_sizes.values.data());
    preparedArgs.push_back(/*offset*/ 0);
    // sizes
    for (auto size : values_and_sizes.sizes) {
      preparedArgs.push_back((void *)size);
    }
    // strides has been removed by serialization
    auto stride = values_and_sizes.length();
    for (auto size : sizes) {
      stride /= size;
      preparedArgs.push_back((void *)stride);
    }
  }
  return outcome::success();
}

outcome::checked<std::unique_ptr<PublicArguments>, StringError>
PublicArguments::unserialize(ClientParameters &clientParameters,
                             std::istream &istream) {
  std::vector<void *> empty;
  std::vector<TensorData> emptyBuffers;
  auto sArguments = std::make_unique<PublicArguments>(
      clientParameters, std::move(empty), std::move(emptyBuffers));
  OUTCOME_TRYV(sArguments->unserializeArgs(istream));
  return std::move(sArguments);
}

outcome::checked<void, StringError>
PublicResult::unserialize(std::istream &istream) {
  for (auto gate : clientParameters.outputs) {
    if (!gate.encryption.hasValue()) {
      return StringError("Clear values are not handled");
    }
    auto lweSize = clientParameters.lweSecretKeyParam(gate).value().lweSize();
    std::vector<int64_t> sizes = gate.shape.dimensions;
    sizes.push_back(lweSize);
    buffers.push_back(unserializeTensorData(sizes, istream));
    if (istream.fail()) {
      return StringError("Cannot read tensor data");
    }
  }
  return outcome::success();
}

outcome::checked<void, StringError>
PublicResult::serialize(std::ostream &ostream) {
  if (incorrectMode(ostream)) {
    return StringError(
        "PublicResult::serialize: ostream should be in binary mode");
  }
  for (auto tensorData : buffers) {
    serializeTensorData(tensorData, ostream);
    if (ostream.fail()) {
      return StringError("Cannot write tensor data");
    }
  }
  return outcome::success();
}

void next_coord_index(size_t index[], size_t sizes[], size_t rank) {
  // increase multi dim index
  for (int r = rank - 1; r >= 0; r--) {
    if (index[r] < sizes[r] - 1) {
      index[r]++;
      return;
    }
    index[r] = 0;
  }
}

size_t global_index(size_t index[], size_t sizes[], size_t strides[],
                    size_t rank) {
  // compute global index from multi dim index
  size_t g_index = 0;
  size_t default_stride = 1;
  for (int r = rank - 1; r >= 0; r--) {
    g_index += index[r] * ((strides[r] == 0) ? default_stride : strides[r]);
    default_stride *= sizes[r];
  }
  return g_index;
}

TensorData tensorDataFromScalar(uint64_t value) { return {{value}, {1}}; }

TensorData tensorDataFromMemRef(size_t memref_rank,
                                encrypted_scalars_t allocated,
                                encrypted_scalars_t aligned, size_t offset,
                                size_t *sizes, size_t *strides) {
  TensorData result;
  assert(aligned != nullptr);
  result.sizes.resize(memref_rank);
  for (size_t r = 0; r < memref_rank; r++) {
    result.sizes[r] = sizes[r];
  }
  // ephemeral multi dim index to compute global strides
  size_t *index = new size_t[memref_rank];
  for (size_t r = 0; r < memref_rank; r++) {
    index[r] = 0;
  }
  auto len = result.length();
  result.values.resize(len);
  // TODO: add a fast path for dense result (no real strides)
  for (size_t i = 0; i < len; i++) {
    int g_index = offset + global_index(index, sizes, strides, memref_rank);
    result.values[i] = aligned[offset + g_index];
    next_coord_index(index, sizes, memref_rank);
  }
  delete[] index;
  // TEMPORARY: That quick and dirty but as this function is used only to
  // convert a result of the mlir program and as data are copied here, we
  // release the alocated pointer if it set.
  if (allocated != nullptr) {
    free(allocated);
  }
  return result;
}

} // namespace clientlib
} // namespace concretelang
