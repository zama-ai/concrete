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
#include "concretelang/ClientLib/ValueDecrypter.h"
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

using concretelang::clientlib::ValueDecrypter;
using concretelang::error::StringError;

class EncryptedArguments;

/// PublicArguments will be sended to the server. It includes encrypted
/// arguments and public keys.
class PublicArguments {
public:
  PublicArguments(const ClientParameters &clientParameters,
                  std::vector<clientlib::SharedScalarOrTensorData> &buffers);
  ~PublicArguments();

  static outcome::checked<std::unique_ptr<PublicArguments>, StringError>
  unserialize(const ClientParameters &expectedParams, std::istream &istream);

  outcome::checked<void, StringError> serialize(std::ostream &ostream);

  std::vector<SharedScalarOrTensorData> &getArguments() { return arguments; }
  ClientParameters &getClientParameters() { return clientParameters; }

  friend class ::concretelang::serverlib::ServerLambda;
  friend class ::mlir::concretelang::JITLambda;

private:
  outcome::checked<void, StringError> unserializeArgs(std::istream &istream);

  ClientParameters clientParameters;
  /// Store buffers of ciphertexts
  std::vector<SharedScalarOrTensorData> arguments;
};

/// PublicResult is a result of a ServerLambda call which contains encrypted
/// results.
struct PublicResult {

  PublicResult(const ClientParameters &clientParameters,
               std::vector<SharedScalarOrTensorData> &&buffers = {})
      : clientParameters(clientParameters), buffers(std::move(buffers)){};

  PublicResult(PublicResult &) = delete;

  /// @brief Return a value from the PublicResult
  /// @param argPos The position of the value in the PublicResult
  /// @return Either the value or an error if there are no value at this
  /// position
  outcome::checked<SharedScalarOrTensorData, StringError>
  getValue(size_t argPos) {
    if (argPos >= buffers.size()) {
      return StringError("result #") << argPos << " does not exists";
    }
    return buffers[argPos];
  }

  /// Create a public result from buffers.
  static std::unique_ptr<PublicResult>
  fromBuffers(const ClientParameters &clientParameters,
              std::vector<SharedScalarOrTensorData> &&buffers) {
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

  /// Get the result at `pos` as a scalar. Decryption happens if the
  /// result is encrypted.
  template <typename T>
  outcome::checked<T, StringError> asClearTextScalar(KeySet &keySet,
                                                     size_t pos) {
    ValueDecrypter decrypter(keySet, clientParameters);
    auto &data = buffers[pos].get();
    return decrypter.template decrypt<T>(data, pos);
  }

  /// Get the result at `pos` as a vector. Decryption happens if the
  /// result is encrypted.
  template <typename T>
  outcome::checked<std::vector<T>, StringError>
  asClearTextVector(KeySet &keySet, size_t pos) {
    ValueDecrypter decrypter(keySet, clientParameters);
    return decrypter.template decryptTensor<T>(buffers[pos].get(), pos);
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
  std::vector<SharedScalarOrTensorData> buffers;
};

/// Helper function to convert from MemRefDescriptor to
/// TensorData
TensorData tensorDataFromMemRef(size_t memref_rank, size_t element_width,
                                bool is_signed, void *allocated, void *aligned,
                                size_t offset, size_t *sizes, size_t *strides);

} // namespace clientlib
} // namespace concretelang

#endif
