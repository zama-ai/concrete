// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <dlfcn.h>

#include "concretelang/ClientLib/ClientLambda.h"
#include "concretelang/ClientLib/Serializers.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

outcome::checked<ClientLambda, StringError>
ClientLambda::load(std::string functionName, std::string jsonPath) {
  OUTCOME_TRY(auto all_params, ClientParameters::load(jsonPath));
  auto param = llvm::find_if(all_params, [&](ClientParameters param) {
    return param.functionName == functionName;
  });

  if (param == all_params.end()) {
    return StringError("ClientLambda: cannot find function ")
           << functionName << " in client parameters" << jsonPath;
  }
  if (param->outputs.size() != 1) {
    return StringError("ClientLambda: output arity (")
           << std::to_string(param->outputs.size()) << ") != 1 is not supprted";
  }

  if (!param->outputs[0].encryption.hasValue()) {
    return StringError("ClientLambda: clear output is not yet supported");
  }
  ClientLambda lambda;
  lambda.clientParameters = *param;
  return lambda;
}

outcome::checked<std::unique_ptr<KeySet>, StringError>
ClientLambda::keySet(std::shared_ptr<KeySetCache> optionalCache,
                     uint64_t seed_msb, uint64_t seed_lsb) {
  return KeySetCache::generate(optionalCache, clientParameters, seed_msb,
                               seed_lsb);
}

outcome::checked<decrypted_scalar_t, StringError>
ClientLambda::decryptReturnedScalar(KeySet &keySet, PublicResult &result) {
  OUTCOME_TRY(auto v, decryptReturnedValues(keySet, result));
  return v[0];
}

outcome::checked<std::vector<decrypted_scalar_t>, StringError>
ClientLambda::decryptReturnedValues(KeySet &keySet, PublicResult &result) {
  return result.asClearTextVector(keySet, 0);
}

outcome::checked<void, StringError> errorResultRank(size_t expected,
                                                    size_t actual) {
  return StringError("Expected result has rank ")
         << expected << " and cannot be converted to rank " << actual;
}

StringError errorIncoherentSizes(size_t flatSize, size_t structuredSize) {
  return StringError("Received ")
         << flatSize << " values but is sizes indicates as global size of "
         << structuredSize;
}

template <typename DecryptedTensor>
DecryptedTensor flatToTensor(decrypted_tensor_1_t &values, size_t *sizes);

template <>
decrypted_tensor_1_t flatToTensor(decrypted_tensor_1_t &values, size_t *sizes) {
  return values;
}

template <>
decrypted_tensor_2_t flatToTensor(decrypted_tensor_1_t &values, size_t *sizes) {
  decrypted_tensor_2_t result(sizes[0]);
  size_t position = 0;
  for (auto &dest0 : result) {
    dest0.resize(sizes[1]);
    for (auto &dest1 : dest0) {
      dest1 = values[position++];
    }
  }
  return result;
}

template <>
decrypted_tensor_3_t flatToTensor(decrypted_tensor_1_t &values, size_t *sizes) {
  decrypted_tensor_3_t result(sizes[0]);
  size_t position = 0;
  for (auto &dest0 : result) {
    dest0.resize(sizes[1]);
    for (auto &dest1 : dest0) {
      dest1.resize(sizes[2]);
      for (auto &dest2 : dest1) {
        dest2 = values[position++];
      }
    }
  }
  return result;
}

template <typename DecryptedTensor>
outcome::checked<DecryptedTensor, StringError>
decryptReturnedTensor(PublicResult &result, ClientLambda &lambda,
                      ClientParameters &params, size_t expectedRank,
                      KeySet &keySet) {
  auto shape = params.outputs[0].shape;
  size_t rank = shape.dimensions.size();
  if (rank != expectedRank) {
    return StringError("Function returns a tensor of rank ")
           << expectedRank << " which cannot be decrypted to rank " << rank;
  }
  OUTCOME_TRY(auto values, lambda.decryptReturnedValues(keySet, result));
  llvm::SmallVector<size_t, 6> sizes;
  for (size_t dim = 0; dim < rank; dim++) {
    sizes.push_back(shape.dimensions[dim]);
  }
  return flatToTensor<DecryptedTensor>(values, sizes.data());
}

outcome::checked<decrypted_tensor_1_t, StringError>
ClientLambda::decryptReturnedTensor1(KeySet &keySet, PublicResult &result) {
  return decryptReturnedTensor<decrypted_tensor_1_t>(
      result, *this, this->clientParameters, 1, keySet);
}

outcome::checked<decrypted_tensor_2_t, StringError>
ClientLambda::decryptReturnedTensor2(KeySet &keySet, PublicResult &result) {
  return decryptReturnedTensor<decrypted_tensor_2_t>(
      result, *this, this->clientParameters, 2, keySet);
}

outcome::checked<decrypted_tensor_3_t, StringError>
ClientLambda::decryptReturnedTensor3(KeySet &keySet, PublicResult &result) {
  return decryptReturnedTensor<decrypted_tensor_3_t>(
      result, *this, this->clientParameters, 3, keySet);
}

template <typename Result>
outcome::checked<Result, StringError>
topLevelDecryptResult(ClientLambda &lambda, KeySet &keySet,
                      PublicResult &result) {
  // compile time error if used
  using COMPATIBLE_RESULT_TYPE = void;
  return (Result)(COMPATIBLE_RESULT_TYPE)0;
}

template <>
outcome::checked<decrypted_scalar_t, StringError>
topLevelDecryptResult<decrypted_scalar_t>(ClientLambda &lambda, KeySet &keySet,
                                          PublicResult &result) {
  return lambda.decryptReturnedScalar(keySet, result);
}

template <>
outcome::checked<decrypted_tensor_1_t, StringError>
topLevelDecryptResult<decrypted_tensor_1_t>(ClientLambda &lambda,
                                            KeySet &keySet,
                                            PublicResult &result) {
  return lambda.decryptReturnedTensor1(keySet, result);
}

template <>
outcome::checked<decrypted_tensor_2_t, StringError>
topLevelDecryptResult<decrypted_tensor_2_t>(ClientLambda &lambda,
                                            KeySet &keySet,
                                            PublicResult &result) {
  return lambda.decryptReturnedTensor2(keySet, result);
}

template <>
outcome::checked<decrypted_tensor_3_t, StringError>
topLevelDecryptResult<decrypted_tensor_3_t>(ClientLambda &lambda,
                                            KeySet &keySet,
                                            PublicResult &result) {
  return lambda.decryptReturnedTensor3(keySet, result);
}

} // namespace clientlib
} // namespace concretelang