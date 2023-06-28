// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_PUBLIC_ARGUMENTS_H
#define CONCRETELANG_CLIENTLIB_PUBLIC_ARGUMENTS_H

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <vector>
#ifdef OUTPUT_COMPRESSION_SUPPORT
#include "compress_lwe/defines.h"
#include "compress_lwe/library.h"
#endif

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EncryptedArguments.h"
#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/ClientLib/ValueDecrypter.h"
#include "concretelang/Common/Error.h"
#include <memory>
#include <variant>

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
  PublicArguments(
      const ClientParameters &clientParameters,
      std::vector<clientlib::SharedScalarOrTensorOrCompressedData> &buffers);

  PublicArguments(const ClientParameters &clientParameters,
                  std::vector<clientlib::ScalarOrTensorData> &&buffers);

  ~PublicArguments();

  static outcome::checked<std::unique_ptr<PublicArguments>, StringError>
  unserialize(const ClientParameters &expectedParams, std::istream &istream);

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

  PublicResult(const ClientParameters &clientParameters)
      : clientParameters(clientParameters), buffers() {}

  PublicResult(const ClientParameters &clientParameters,
               std::optional<EvaluationKeys> &evaluationKeys,
               std::vector<ScalarOrTensorData> &&buffers_)
      : clientParameters(clientParameters), buffers() {

    assert(buffers_.size() == clientParameters.outputs.size());

    assert(buffers_.size() == 1);

    assert(clientParameters.outputs.size() <= 1);

    if (clientParameters.outputs.size() == 1) {

      auto &gate = clientParameters.outputs[0];
      if (gate.encryption.has_value() && gate.compression) {
#ifdef OUTPUT_COMPRESSION_SUPPORT
        assert(std::holds_alternative<TensorData>(buffers_[0]));

        TensorData tensor = std::get<TensorData>(std::move(buffers_[0]));

        auto dimensions = tensor.getDimensions();

        u_int64_t dim_product = 1;

        for (u_int64_t i = 0; i < dimensions.size() - 1; i++) {
          dim_product *= dimensions[i];
        }

        uint lwe_dim = dimensions[dimensions.size() - 1] - 1;

        assert(evaluationKeys.has_value());

        auto &opt_comp_key = evaluationKeys->getCompressionKey();

        assert(opt_comp_key.has_value());

        comp::CompressionKey comp_key = *opt_comp_key;

        buffers.push_back(std::make_shared<ScalarOrTensorOrCompressedData>(
            std::make_shared<comp::CompressedCiphertext>(compressBatched(
                comp_key, tensor.getElementPointer<u_int64_t>(0), lwe_dim,
                dim_product))));
#else
        // Compression not supported
        abort();
#endif
      } else {

        if (std::holds_alternative<ScalarData>(buffers_[0])) {
          buffers.push_back(std::make_shared<ScalarOrTensorOrCompressedData>(
              std::get<ScalarData>(buffers_[0])));
        } else {
          buffers.push_back(std::make_shared<ScalarOrTensorOrCompressedData>(
              std::get<TensorData>(std::move(buffers_[0]))));
        }
      }
    }
  }

  PublicResult(PublicResult &) = delete;

  /// @brief Return a value from the PublicResult
  /// @param argPos The position of the value in the PublicResult
  /// @return Either the value or an error if there are no value at this
  /// position
  outcome::checked<SharedScalarOrTensorOrCompressedData, StringError>
  getValue(size_t argPos) {
    if (argPos >= buffers.size()) {
      return StringError("result #") << argPos << " does not exists";
    }
    return buffers[argPos];
  }

  /// Create a public result from buffers.
  static std::unique_ptr<PublicResult>
  fromBuffers(const ClientParameters &clientParameters,
              std::optional<EvaluationKeys> &evaluationKeys,
              std::vector<ScalarOrTensorData> &&buffers) {
    return std::make_unique<PublicResult>(clientParameters, evaluationKeys,
                                          std::move(buffers));
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

    OUTCOME_TRY(auto result, decrypter.decryptValues(buffers[pos].get(), pos));
    return (T)result[0];
  }

  /// Get the result at `pos` as a vector. Decryption happens if the
  /// result is encrypted.
  template <typename T>
  outcome::checked<std::vector<T>, StringError>
  asClearTextVector(KeySet &keySet, size_t pos) {
    ValueDecrypter decrypter(keySet, clientParameters);

    OUTCOME_TRY(std::vector<uint64_t> result2,
                decrypter.decryptValues(buffers[pos].get(), pos));

    std::vector<T> result;
    for (auto &a : result2) {
      result.push_back((T)a);
    }
    return result;
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
  std::vector<SharedScalarOrTensorOrCompressedData> buffers;
};

/// Helper function to convert from MemRefDescriptor to
/// TensorData
TensorData tensorDataFromMemRef(size_t memref_rank, size_t element_width,
                                bool is_signed, void *allocated, void *aligned,
                                size_t offset, size_t *sizes, size_t *strides);

uint64_t decode_1padded_integer(uint64_t decrypted, uint64_t precision,
                                bool signe);

uint64_t decode_crt(std::vector<int64_t> &decrypted_remainders,
                    std::vector<int64_t> &crt_bases, bool signe);

} // namespace clientlib
} // namespace concretelang

#endif
