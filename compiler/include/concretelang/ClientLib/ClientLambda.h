// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_CLIENT_LAMBDA_H
#define CONCRETELANG_CLIENTLIB_CLIENT_LAMBDA_H

#include <cassert>

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EncryptedArgs.h"
#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/ClientLib/KeySetCache.h"
#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/Common/Error.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;
using scalar_in = uint8_t;
using scalar_out = uint64_t;
using tensor1_in = std::vector<scalar_in>;
using tensor2_in = std::vector<std::vector<scalar_in>>;
using tensor3_in = std::vector<std::vector<std::vector<scalar_in>>>;
using tensor1_out = std::vector<scalar_out>;
using tensor2_out = std::vector<std::vector<scalar_out>>;
using tensor3_out = std::vector<std::vector<std::vector<scalar_out>>>;

class ClientLambda {
  /// Low-level class to create the client side view of a FHE function.
public:
  virtual ~ClientLambda() = default;
  static outcome::checked<ClientLambda, StringError>
  /// Construct a ClientLambda from a ClientParameter file.
  load(std::string funcName, std::string jsonPath);

  /// Emit a call to the given ostream, no meta-date are include, so it's the
  /// responsability of the the caller/callee to verify the add/verify the
  /// function to be called.
  outcome::checked<void, StringError>
  untypedSerializeCall(PublicArguments &publicArguments, std::ostream &ostream);

  /// Generate or get from cache a KeySet suitable for this ClientLambda
  outcome::checked<std::unique_ptr<KeySet>, StringError>
  keySet(std::shared_ptr<KeySetCache> optionalCache, uint64_t seed_msb,
         uint64_t seed_lsb);

  outcome::checked<std::vector<decrypted_scalar_t>, StringError>
  decryptReturnedValues(KeySet &keySet, std::istream &istream);

  outcome::checked<decrypted_scalar_t, StringError>
  decryptReturnedScalar(KeySet &keySet, std::istream &istream);

  outcome::checked<decrypted_tensor_1_t, StringError>
  decryptReturnedTensor1(KeySet &keySet, std::istream &istream);

  outcome::checked<decrypted_tensor_2_t, StringError>
  decryptReturnedTensor2(KeySet &keySet, std::istream &istream);

  outcome::checked<decrypted_tensor_3_t, StringError>
  decryptReturnedTensor3(KeySet &keySet, std::istream &istream);

public:
  ClientParameters clientParameters;
};

template <typename Result>
outcome::checked<Result, StringError>
topLevelDecryptResult(ClientLambda &lambda, KeySet &keySet,
                      std::istream &istream);

template <typename Result, typename... Args>
class TypedClientLambda : public ClientLambda {

public:
  static outcome::checked<TypedClientLambda<Result, Args...>, StringError>
  load(std::string funcName, std::string jsonPath) {
    OUTCOME_TRY(auto lambda, ClientLambda::load(funcName, jsonPath));
    return TypedClientLambda(lambda);
  }

  /// Emit a call on this lambda to a binary ostream.
  /// The ostream is responsible for transporting the call to a
  /// ServerLambda::real_call_write function. ostream must be in binary mode
  /// std::ios_base::openmode::binary
  outcome::checked<void, StringError>
  serializeCall(Args... args, std::shared_ptr<KeySet> keySet,
                std::ostream &ostream) {
    OUTCOME_TRY(auto publicArguments, publicArguments(args..., keySet));
    return ClientLambda::untypedSerializeCall(publicArguments, ostream);
  }

  outcome::checked<PublicArguments, StringError>
  publicArguments(Args... args, std::shared_ptr<KeySet> keySet) {
    OUTCOME_TRY(auto clientArguments, EncryptedArgs::create(keySet, args...));
    return clientArguments->asPublicArguments(clientParameters,
                                              keySet->runtimeContext());
  }

  outcome::checked<Result, StringError> decryptReturned(KeySet &keySet,
                                                        std::istream &istream) {
    return topLevelDecryptResult<Result>((*this), keySet, istream);
  }

  TypedClientLambda(ClientLambda &lambda) : ClientLambda(lambda) {
    // TODO: check parameter types
    // TODO: add static check on types vs lambda inputs/outpus
  }

protected:
  // Workaround, gcc 6 does not support partial template specialisation in class
  template <typename Result_>
  friend outcome::checked<Result_, StringError>
  topLevelDecryptResult(ClientLambda &lambda, KeySet &keySet,
                        std::istream &istream);
};

template <>
outcome::checked<decrypted_scalar_t, StringError>
topLevelDecryptResult<decrypted_scalar_t>(ClientLambda &lambda, KeySet &keySet,
                                          std::istream &istream);

template <>
outcome::checked<decrypted_tensor_1_t, StringError>
topLevelDecryptResult<decrypted_tensor_1_t>(ClientLambda &lambda,
                                            KeySet &keySet,
                                            std::istream &istream);

template <>
outcome::checked<decrypted_tensor_2_t, StringError>
topLevelDecryptResult<decrypted_tensor_2_t>(ClientLambda &lambda,
                                            KeySet &keySet,
                                            std::istream &istream);

} // namespace clientlib
} // namespace concretelang

#endif
