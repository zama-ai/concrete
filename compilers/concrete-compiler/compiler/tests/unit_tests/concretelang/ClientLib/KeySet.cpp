#include <gtest/gtest.h>

#include "concrete/curves.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EncryptedArguments.h"
#include "concretelang/ClientLib/EvaluationKeys.h"
#include "tests_tools/assert.h"

namespace clientlib = concretelang::clientlib;

// Define a fixture for instantiate test with client parameters
class KeySetTest
    : public ::testing::TestWithParam<clientlib::ClientParameters> {
protected:
  clientlib::ClientParameters clientParameters;
};

// Test case encrypt and decrypt
TEST_P(KeySetTest, encrypt_decrypt) {

  auto clientParameters = GetParam();

  __uint128_t seed = 0;

  // Generate the client keySet
  ASSERT_ASSIGN_OUTCOME_VALUE(
      keySet,
      clientlib::KeySet::generate(
          clientParameters, concretelang::clientlib::ConcreteCSPRNG(seed)));

  // Allocate the ciphertext
  uint64_t *ciphertext = nullptr;
  uint64_t size = 0;
  ASSERT_OUTCOME_HAS_VALUE(keySet->allocate_lwe(0, &ciphertext, size));

  // Encrypt
  uint64_t input = 0;
  ASSERT_OUTCOME_HAS_VALUE(keySet->encrypt_lwe(0, ciphertext, input));

  // Decrypt
  uint64_t output;
  ASSERT_OUTCOME_HAS_VALUE(keySet->decrypt_lwe(0, ciphertext, output));

  ASSERT_EQ(input, output) << "decrypted value differs than the encrypted one";
}

///////////////////////////////////////////////////////////////////////////////
/// Instantiate test suite with generated client parameters ///////////////////
///////////////////////////////////////////////////////////////////////////////

/// Create a client parameters with just one secret key of `dimension` and with
/// one input scalar gate and one output scalar gate on the same key
clientlib::ClientParameters generateClientParameterOneScalarOneScalar(
    clientlib::LweDimension dimension, clientlib::Precision precision,
    clientlib::CRTDecomposition crtDecomposition) {
  // One secret key with the given dimension
  clientlib::ClientParameters params;
  params.secretKeys.push_back({/*.dimension =*/dimension});
  // One input and output encryption gate on the same secret key and encoded
  // with the same precision
  const auto v0Curve = concrete::getSecurityCurve(128, concrete::BINARY);

  clientlib::EncryptionGate encryption;
  encryption.secretKeyID = clientlib::BIG_KEY;
  encryption.encoding.precision = precision;
  encryption.encoding.crt = crtDecomposition;
  encryption.variance = v0Curve->getVariance(1, dimension, 64);
  clientlib::CircuitGate gate;
  gate.encryption = encryption;
  params.inputs.push_back(gate);
  params.outputs.push_back(gate);
  return params;
}

std::vector<clientlib::ClientParameters> generateAllParameters() {
  // All lwe dimensions to test
  std::vector<clientlib::LweDimension> lweDimensions{
      1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13,
  };

  // All precision to test
  std::vector<clientlib::Precision> precisions(8, 0);
  llvm::for_each(llvm::enumerate(precisions),
                 [](auto p) { p.value() = p.index() + 1; });

  // All crt decomposition to test
  std::vector<clientlib::CRTDecomposition> crtDecompositions{
      // Empty crt decompositon means no decomposition
      {},
      // The default decomposition for 16 bits
      {7, 8, 9, 11, 13},
  };

  // All client parameters to test
  std::vector<clientlib::ClientParameters> parameters;

  for (auto dimension : lweDimensions) {
    for (auto precision : precisions) {
      for (auto crtDecomposition : crtDecompositions) {
        // Do not use dimension 512 for precision 8
        if (precision > 7 && dimension < (1 << 10))
          continue;
        parameters.push_back(generateClientParameterOneScalarOneScalar(
            dimension, precision, crtDecomposition));
      }
    }
  }

  return parameters;
}

INSTANTIATE_TEST_SUITE_P(
    OneScalarOnScalar, KeySetTest, ::testing::ValuesIn(generateAllParameters()),
    [](const testing::TestParamInfo<clientlib::ClientParameters> info) {
      auto cp = info.param;
      auto input_0 = cp.inputs[0];
      auto paramDescription =
          std::string("lweDimension_") +
          std::to_string(cp.lweSecretKeyParam(input_0).value().dimension) +
          "_precision_" +
          std::to_string(input_0.encryption.value().encoding.precision);
      auto crt = input_0.encryption.value().encoding.crt;
      if (!crt.empty()) {
        paramDescription = paramDescription + "_crt_";
        for (auto b : crt) {
          paramDescription = paramDescription + "_" + std::to_string(b);
        }
      }
      return paramDescription;
    });
