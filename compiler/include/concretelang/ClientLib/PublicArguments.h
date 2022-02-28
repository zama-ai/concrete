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
namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

class EncryptedArguments;
class PublicArguments {
  /// PublicArguments will be sended to the server. It includes encrypted
  /// arguments and public keys.
public:
  PublicArguments(
      const ClientParameters &clientParameters, RuntimeContext runtimeContext,
      bool clearRuntimeContext, std::vector<void *> &&preparedArgs,
      std::vector<encrypted_scalars_and_sizes_t> &&ciphertextBuffers);
  ~PublicArguments();
  PublicArguments(PublicArguments &other) = delete;
  PublicArguments(PublicArguments &&other) = delete;

  static outcome::checked<std::shared_ptr<PublicArguments>, StringError>
  unserialize(ClientParameters &expectedParams, std::istream &istream);

  outcome::checked<void, StringError> serialize(std::ostream &ostream);

private:
  friend class ::concretelang::serverlib::ServerLambda; // from ServerLib

  outcome::checked<void, StringError> unserializeArgs(std::istream &istream);

  ClientParameters clientParameters;
  RuntimeContext runtimeContext;
  std::vector<void *> preparedArgs;
  // Store buffers of ciphertexts
  std::vector<encrypted_scalars_and_sizes_t> ciphertextBuffers;

  // Indicates if this public argument own the runtime keys.
  bool clearRuntimeContext;
};

struct PublicResult {
  /// PublicResult is a result of a ServerLambda call which contains encrypted
  /// results.

  PublicResult(const ClientParameters &clientParameters,
               std::vector<encrypted_scalars_and_sizes_t> buffers = {})
      : clientParameters(clientParameters), buffers(buffers){};

  PublicResult(PublicResult &) = delete;

  /// Create a public result from buffers.
  static std::unique_ptr<PublicResult>
  fromBuffers(const ClientParameters &clientParameters,
              std::vector<encrypted_scalars_and_sizes_t> buffers) {
    return std::make_unique<PublicResult>(clientParameters, buffers);
  }

  /// Unserialize from a input stream.
  outcome::checked<void, StringError> unserialize(std::istream &istream);

  /// Serialize into an output stream.
  outcome::checked<void, StringError> serialize(std::ostream &ostream);

  /// Decrypt the result at `pos` as a vector.
  outcome::checked<std::vector<decrypted_scalar_t>, StringError>
  decryptVector(KeySet &keySet, size_t pos);

private:
  friend class ::concretelang::serverlib::ServerLambda;
  ClientParameters clientParameters;
  std::vector<encrypted_scalars_and_sizes_t> buffers;
};

} // namespace clientlib
} // namespace concretelang

#endif
