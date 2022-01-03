// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
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
PublicArguments::PublicArguments(
    const ClientParameters &clientParameters, RuntimeContext runtimeContext,
    bool clearRuntimeContext, std::vector<void *> &&preparedArgs_,
    std::vector<encrypted_scalars_and_sizes_t> &&ciphertextBuffers_)
    : clientParameters(clientParameters), runtimeContext(runtimeContext),
      clearRuntimeContext(clearRuntimeContext) {
  preparedArgs = std::move(preparedArgs_);
  ciphertextBuffers = std::move(ciphertextBuffers_);
}

PublicArguments::PublicArguments(PublicArguments &&other) {
  clientParameters = other.clientParameters;
  runtimeContext = other.runtimeContext;
  runtimeContext.bsk = std::move(other.runtimeContext.bsk);
  clearRuntimeContext = other.clearRuntimeContext;
  preparedArgs = std::move(other.preparedArgs);
  ciphertextBuffers = std::move(other.ciphertextBuffers);
  // transfer ownership
  other.clearRuntimeContext = false;
  other.runtimeContext.ksk = nullptr;
}

PublicArguments::~PublicArguments() {
  if (!clearRuntimeContext) {
    return;
  }
  for (auto bsk_entry : runtimeContext.bsk) {
    free_lwe_bootstrap_key_u64(bsk_entry.second);
  }
  runtimeContext.bsk.clear();
  if (runtimeContext.ksk != nullptr) {
    free_lwe_keyswitch_key_u64(runtimeContext.ksk);
    runtimeContext.ksk = nullptr;
  }
}

outcome::checked<void, StringError>
PublicArguments::serialize(std::ostream &ostream) {
  if (incorrectMode(ostream)) {
    return StringError(
        "PublicArguments::serialize: ostream should be in binary mode");
  }
  ostream << runtimeContext;
  size_t iPreparedArgs = 0;
  int iGate = -1;
  for (auto gate : clientParameters.inputs) {
    iGate++;
    size_t rank = gate.shape.dimensions.size();
    if (!gate.encryption.hasValue()) {
      return StringError("PublicArguments::serialize: Clear arguments "
                         "are not supported. Argument ")
             << iGate;
    }
    /*auto allocated = */ preparedArgs[iPreparedArgs++];
    auto aligned = (encrypted_scalars_t)preparedArgs[iPreparedArgs++];
    assert(aligned != nullptr);
    auto offset = (size_t)preparedArgs[iPreparedArgs++];
    std::vector<size_t> sizes; // includes lweSize as last dim
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
    serializeEncryptedValues(sizes, values, ostream);
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
    auto lweSize = clientParameters.lweSecretKeyParam(gate).lweSize();
    std::vector<int64_t> sizes = gate.shape.dimensions;
    sizes.push_back(lweSize);
    ciphertextBuffers.push_back(
        std::move(unserializeEncryptedValues(sizes, istream)));
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

outcome::checked<std::shared_ptr<PublicArguments>, StringError>
PublicArguments::unserialize(ClientParameters &clientParameters,
                             std::istream &istream) {
  RuntimeContext runtimeContext;
  istream >> runtimeContext;
  if (istream.fail()) {
    return StringError("Cannot read runtime context");
  }
  std::vector<void *> empty;
  std::vector<encrypted_scalars_and_sizes_t> emptyBuffers;
  // On server side the PublicArguments is responsible for the context
  auto clearRuntimeContext = true;
  auto sArguments = std::make_shared<PublicArguments>(
      clientParameters, runtimeContext, clearRuntimeContext, std::move(empty),
      std::move(emptyBuffers));
  OUTCOME_TRYV(sArguments->unserializeArgs(istream));
  sArguments->preparedArgs.push_back((void *)&runtimeContext);
  return sArguments;
}

} // namespace clientlib
} // namespace concretelang