// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_PUBLIC_ARGUMENTS_H
#define CONCRETELANG_CLIENTLIB_PUBLIC_ARGUMENTS_H

#include <iostream>

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EncryptedArgs.h"
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

class EncryptedArgs;
class PublicArguments {
  /// PublicArguments will be sended to the server. It includes encrypted
  /// arguments and public keys.
public:
  PublicArguments(
      const ClientParameters &clientParameters, RuntimeContext runtimeContext,
      bool clearRuntimeContext, std::vector<void *> &&preparedArgs,
      std::vector<encrypted_scalars_and_sizes_t> &&ciphertextBuffers);
  PublicArguments(PublicArguments &other) = delete;
  // to have proper owership transfer (outcome and local object)
  PublicArguments(PublicArguments &&other);
  ~PublicArguments();

  void freeIfNotOwned(std::vector<encrypted_scalar_t> res);

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
  bool clearRuntimeContext;
};

} // namespace clientlib
} // namespace concretelang

#endif
