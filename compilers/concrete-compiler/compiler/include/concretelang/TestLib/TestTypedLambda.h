// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TESTLIB_TEST_TYPED_LAMBDA_H
#define CONCRETELANG_TESTLIB_TEST_TYPED_LAMBDA_H

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientLambda.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/KeySetCache.h"
#include "concretelang/ClientLib/Serializers.h"
#include "concretelang/Common/Error.h"
#include "concretelang/ServerLib/ServerLambda.h"

namespace concretelang {
namespace testlib {

using concretelang::clientlib::ClientLambda;
using concretelang::clientlib::ClientParameters;
using concretelang::clientlib::KeySet;
using concretelang::clientlib::KeySetCache;
using concretelang::error::StringError;
using concretelang::serverlib::ServerLambda;

inline void freeStringMemory(std::string &s) {
  std::string empty;
  s.swap(empty);
}

template <typename Result, typename... Args>
class TestTypedLambda
    : public concretelang::clientlib::TypedClientLambda<Result, Args...> {

  template <typename Result_, typename... Args_>
  using TypedClientLambda =
      concretelang::clientlib::TypedClientLambda<Result_, Args_...>;

public:
  static outcome::checked<TestTypedLambda, StringError>
  load(std::string funcName, std::string outputLib, uint64_t seed_msb = 0,
       uint64_t seed_lsb = 0,
       std::shared_ptr<KeySetCache> unsecure_cache = nullptr) {
    std::string jsonPath =
        mlir::concretelang::CompilerEngine::Library::getProgramInfoPath(
            outputLib);
    OUTCOME_TRY(auto cLambda, ClientLambda::load(funcName, jsonPath));
    OUTCOME_TRY(auto sLambda, ServerLambda::load(funcName, outputLib));
    OUTCOME_TRY(std::shared_ptr<KeySet> keySet,
                KeySetCache::generate(unsecure_cache, cLambda.clientParameters,
                                      seed_msb, seed_lsb));
    return TestTypedLambda(cLambda, sLambda, keySet);
  }

  TestTypedLambda(ClientLambda &cLambda, ServerLambda &sLambda,
                  std::shared_ptr<KeySet> keySet)
      : TypedClientLambda<Result, Args...>(cLambda), serverLambda(sLambda),
        keySet(keySet) {}

  TestTypedLambda(TypedClientLambda<Result, Args...> &cLambda,
                  ServerLambda &sLambda, std::shared_ptr<KeySet> keySet)
      : TypedClientLambda<Result, Args...>(cLambda), serverLambda(sLambda),
        keySet(keySet) {}

  outcome::checked<Result, StringError> call(Args... args) {
    // std::string message;

    // client stream
    // std::ostringstream clientOuput(std::ios::binary);
    // client argument encryption
    OUTCOME_TRY(auto encryptedArgs,
                clientlib::EncryptedArguments::create(*keySet, args...));
    OUTCOME_TRY(auto publicArgument,
                encryptedArgs->exportPublicArguments(this->clientParameters));
    // client argument serialization
    // publicArgument->serialize(clientOuput);
    // message = clientOuput.str();

    // server stream
    // std::istringstream serverInput(message, std::ios::binary);
    // freeStringMemory(message);
    //
    // OUTCOME_TRY(auto publicArguments,
    // clientlib::PublicArguments::unserialize(
    //                                      this->clientParameters,
    //                                      serverInput));

    // server function call
    auto evaluationKeys = keySet->evaluationKeys();
    auto publicResult = serverLambda.call(*publicArgument, evaluationKeys);
    if (!publicResult) {
      return StringError("failed calling function");
    }

    // client result decryption
    return this->decryptResult(*keySet, *(publicResult.get()));
  }

private:
  ServerLambda serverLambda;
  std::shared_ptr<KeySet> keySet;
};

template <typename Result, typename... Args>
static TestTypedLambda<Result, Args...> TestTypedLambdaFrom(
    concretelang::clientlib::TypedClientLambda<Result, Args...> &cLambda,
    ServerLambda &sLambda, std::shared_ptr<KeySet> keySet) {
  return TestTypedLambda<Result, Args...>(cLambda, sLambda, keySet);
}

} // namespace testlib
} // namespace concretelang

#endif
