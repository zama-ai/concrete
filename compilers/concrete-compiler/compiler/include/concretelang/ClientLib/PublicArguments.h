// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_PUBLIC_ARGUMENTS_H
#define CONCRETELANG_CLIENTLIB_PUBLIC_ARGUMENTS_H

#include <iostream>

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EncryptedArguments.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/Common/Error.h"

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

/// PublicArguments will be sended to the server. It includes encrypted
/// arguments and public keys.
class PublicArguments {
public:
  PublicArguments(const ClientParameters &clientParameters,
                  std::vector<ScalarOrTensorData> &&ciphertextBuffers);
  ~PublicArguments();
  PublicArguments(PublicArguments &other) = delete;
  PublicArguments(PublicArguments &&other) = delete;

  static outcome::checked<std::unique_ptr<PublicArguments>, StringError>
  unserialize(ClientParameters &expectedParams, std::istream &istream);

  outcome::checked<void, StringError> serialize(std::ostream &ostream);

  std::vector<ScalarOrTensorData> &getArguments() { return arguments; }
  ClientParameters &getClientParameters() { return clientParameters; }

  friend class ::concretelang::serverlib::ServerLambda;
  friend class ::mlir::concretelang::JITLambda;

private:
  outcome::checked<void, StringError> unserializeArgs(std::istream &istream);

  ClientParameters clientParameters;
  /// Store buffers of ciphertexts
  std::vector<ScalarOrTensorData> arguments;
};

/// PublicResult is a result of a ServerLambda call which contains encrypted
/// results.
struct PublicResult {

  PublicResult(const ClientParameters &clientParameters,
               std::vector<ScalarOrTensorData> &&buffers = {})
      : clientParameters(clientParameters), buffers(std::move(buffers)){};

  PublicResult(PublicResult &) = delete;

  /// Create a public result from buffers.
  static std::unique_ptr<PublicResult>
  fromBuffers(const ClientParameters &clientParameters,
              std::vector<ScalarOrTensorData> &&buffers) {
    return std::make_unique<PublicResult>(clientParameters, std::move(buffers));
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

  /// Get the original integer that was decomposed into chunks of `chunkWidth`
  /// bits each
  uint64_t fromChunks(std::vector<uint64_t> chunks, unsigned int chunkWidth) {
    uint64_t value = 0;
    uint64_t mask = (1 << chunkWidth) - 1;
    for (size_t i = 0; i < chunks.size(); i++) {
      auto chunk = chunks[i] & mask;
      value += chunk << (chunkWidth * i);
    }
    return value;
  }

  /// Get the result at `pos` as a scalar. Decryption happens if the
  /// result is encrypted.
  template <typename T>
  outcome::checked<T, StringError> asClearTextScalar(KeySet &keySet,
                                                     size_t pos) {
    OUTCOME_TRY(auto gate, clientParameters.ouput(pos));
    if (!gate.isEncrypted())
      return buffers[pos].getScalar().getValue<T>();

    // Chunked integers are represented as tensors at a lower level, so we need
    // to deal with them as tensors, then build the resulting scalar out of the
    // tensor values
    if (gate.chunkInfo.has_value()) {
      OUTCOME_TRY(std::vector<uint64_t> decryptedChunks,
                  this->asClearTextVector<uint64_t>(keySet, pos));
      uint64_t decrypted = fromChunks(decryptedChunks, gate.chunkInfo->width);
      return (T)decrypted;
    }

    auto &buffer = buffers[pos].getTensor();

    auto ciphertext = buffer.getOpaqueElementPointer(0);
    uint64_t decrypted;

    // Convert to uint64_t* as required by `KeySet::decrypt_lwe`
    // FIXME: this may break alignment restrictions on some
    // architectures
    auto ciphertextu64 = reinterpret_cast<uint64_t *>(ciphertext);
    OUTCOME_TRYV(keySet.decrypt_lwe(0, ciphertextu64, decrypted));

    return (T)decrypted;
  }

  /// Get the result at `pos` as a vector. Decryption happens if the
  /// result is encrypted.
  template <typename T>
  outcome::checked<std::vector<T>, StringError>
  asClearTextVector(KeySet &keySet, size_t pos) {
    OUTCOME_TRY(auto gate, clientParameters.ouput(pos));
    if (!gate.isEncrypted())
      return buffers[pos].getTensor().asFlatVector<T>();

    auto &buffer = buffers[pos].getTensor();
    auto lweSize = clientParameters.lweBufferSize(gate);

    std::vector<T> decryptedValues(buffer.length() / lweSize);
    for (size_t i = 0; i < decryptedValues.size(); i++) {
      auto ciphertext = buffer.getOpaqueElementPointer(i * lweSize);
      uint64_t decrypted;

      // Convert to uint64_t* as required by `KeySet::decrypt_lwe`
      // FIXME: this may break alignment restrictions on some
      // architectures
      auto ciphertextu64 = reinterpret_cast<uint64_t *>(ciphertext);
      OUTCOME_TRYV(keySet.decrypt_lwe(0, ciphertextu64, decrypted));
      decryptedValues[i] = decrypted;
    }
    return decryptedValues;
  }

  /// Return the shape of the clear tensor of a result.
  outcome::checked<std::vector<int64_t>, StringError>
  asClearTextShape(size_t pos) {
    OUTCOME_TRY(auto gate, clientParameters.ouput(pos));
    return gate.shape.dimensions;
  }

  // private: TODO tmp
  friend class ::concretelang::serverlib::ServerLambda;
  ClientParameters clientParameters;
  std::vector<ScalarOrTensorData> buffers;
};

/// Helper function to convert from MemRefDescriptor to
/// TensorData
TensorData tensorDataFromMemRef(size_t memref_rank, size_t element_width,
                                bool is_signed, void *allocated, void *aligned,
                                size_t offset, size_t *sizes, size_t *strides);

} // namespace clientlib
} // namespace concretelang

#endif
