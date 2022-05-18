// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_PUBLIC_ARGUMENTS_H
#define CONCRETELANG_CLIENTLIB_PUBLIC_ARGUMENTS_H

#include <iostream>

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EncryptedArguments.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Runtime/context.h"

namespace concretelang {
namespace serverlib {
class ServerLambda;
}
} // namespace concretelang
namespace mlir {
namespace concretelang {
class JITLambda;
}
} // namespace mlir
namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

class EncryptedArguments;
class PublicArguments {
  /// PublicArguments will be sended to the server. It includes encrypted
  /// arguments and public keys.
public:
  PublicArguments(const ClientParameters &clientParameters,
                  std::vector<void *> &&preparedArgs,
                  std::vector<TensorData> &&ciphertextBuffers);
  ~PublicArguments();
  PublicArguments(PublicArguments &other) = delete;
  PublicArguments(PublicArguments &&other) = delete;

  static outcome::checked<std::unique_ptr<PublicArguments>, StringError>
  unserialize(ClientParameters &expectedParams, std::istream &istream);

  outcome::checked<void, StringError> serialize(std::ostream &ostream);

private:
  friend class ::concretelang::serverlib::ServerLambda;
  friend class ::mlir::concretelang::JITLambda;

  outcome::checked<void, StringError> unserializeArgs(std::istream &istream);

  ClientParameters clientParameters;
  std::vector<void *> preparedArgs;
  // Store buffers of ciphertexts
  std::vector<TensorData> ciphertextBuffers;
};

struct PublicResult {
  /// PublicResult is a result of a ServerLambda call which contains encrypted
  /// results.

  PublicResult(const ClientParameters &clientParameters,
               std::vector<TensorData> buffers = {})
      : clientParameters(clientParameters), buffers(buffers){};

  PublicResult(PublicResult &) = delete;

  /// Create a public result from buffers.
  static std::unique_ptr<PublicResult>
  fromBuffers(const ClientParameters &clientParameters,
              std::vector<TensorData> buffers) {
    return std::make_unique<PublicResult>(clientParameters, buffers);
  }

  /// Unserialize from an input stream inplace.
  outcome::checked<void, StringError> unserialize(std::istream &istream);
  /// Unserialize from an input stream returning a new PublicResult.
  static outcome::checked<std::unique_ptr<PublicResult>, StringError>
  unserialize(ClientParameters &expectedParams, std::istream &istream) {
    auto publicResult = std::make_unique<PublicResult>(expectedParams);
    OUTCOME_TRYV(publicResult->unserialize(istream));
    return std::move(publicResult);
  }
  /// Serialize into an output stream.
  outcome::checked<void, StringError> serialize(std::ostream &ostream);

  /// Get the result at `pos` as a vector, if the result is a scalar returns a
  /// vector of size 1. Decryption happens if the result is encrypted.
  // outcome::checked<std::vector<decrypted_scalar_t>, StringError>
  // asClearTextVector(KeySet &keySet, size_t pos);

  template <typename T>
  outcome::checked<std::vector<T>, StringError>
  asClearTextVector(KeySet &keySet, size_t pos) {
    OUTCOME_TRY(auto gate, clientParameters.ouput(pos));
    if (!gate.isEncrypted()) {
      std::vector<T> result;
      result.reserve(buffers[pos].values.size());
      std::copy(buffers[pos].values.begin(), buffers[pos].values.end(),
                std::back_inserter(result));
      return result;
    }

    auto buffer = buffers[pos];
    auto lweSize = clientParameters.lweSecretKeyParam(gate).value().lweSize();

    std::vector<T> decryptedValues(buffer.length() / lweSize);
    for (size_t i = 0; i < decryptedValues.size(); i++) {
      auto ciphertext = &buffer.values[i * lweSize];
      uint64_t decrypted;
      OUTCOME_TRYV(keySet.decrypt_lwe(0, ciphertext, decrypted));
      decryptedValues[i] = decrypted;
    }
    return decryptedValues;
  }

  // private: TODO tmp
  friend class ::concretelang::serverlib::ServerLambda;
  ClientParameters clientParameters;
  std::vector<TensorData> buffers;
};

/// Helper function to convert from a scalar to TensorData
TensorData tensorDataFromScalar(uint64_t value);

/// Helper function to convert from MemRefDescriptor to
/// TensorData
TensorData tensorDataFromMemRef(size_t memref_rank,
                                encrypted_scalars_t allocated,
                                encrypted_scalars_t aligned, size_t offset,
                                size_t *sizes, size_t *strides);

} // namespace clientlib
} // namespace concretelang

#endif
