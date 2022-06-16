
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "../fixture/EndToEndFixture.h"
#include "../tests_tools/keySetCache.h"
#include "concretelang/Support/JITSupport.h"
#include "concretelang/Support/LibrarySupport.h"
#include "end_to_end_jit_test.h"

template <typename LambdaSupport>
void compile_and_run(EndToEndDesc desc, LambdaSupport support) {

  /* 1 - Compile the program */
  auto compilationResult = support.compile(desc.program);
  ASSERT_EXPECTED_SUCCESS(compilationResult);

  /* 2 - Load the client parameters and build the keySet */
  auto clientParameters = support.loadClientParameters(**compilationResult);
  ASSERT_EXPECTED_SUCCESS(clientParameters);

  auto keySet = support.keySet(*clientParameters, getTestKeySetCache());
  ASSERT_EXPECTED_SUCCESS(keySet);

  auto evaluationKeys = (*keySet)->evaluationKeys();

  /* 3 - Load the server lambda */
  auto serverLambda = support.loadServerLambda(**compilationResult);
  ASSERT_EXPECTED_SUCCESS(serverLambda);

  /* For each test entries */
  for (auto test : desc.tests) {
    std::vector<mlir::concretelang::LambdaArgument *> inputArguments;
    inputArguments.reserve(test.inputs.size());
    for (auto input : test.inputs) {
      auto arg = valueDescriptionToLambdaArgument(input);
      ASSERT_EXPECTED_SUCCESS(arg);
      inputArguments.push_back(arg.get());
    }

    /* 4 - Create the public arguments */
    auto publicArguments =
        support.exportArguments(*clientParameters, **keySet, inputArguments);
    ASSERT_EXPECTED_SUCCESS(publicArguments);

    /* 5 - Call the server lambda */
    auto publicResult =
        support.serverCall(*serverLambda, **publicArguments, evaluationKeys);
    ASSERT_EXPECTED_SUCCESS(publicResult);

    /* 6 - Decrypt the public result */
    auto result = mlir::concretelang::typedResult<
        std::unique_ptr<mlir::concretelang::LambdaArgument>>(**keySet,
                                                             **publicResult);

    /* 7 - Check result */
    ASSERT_EXPECTED_SUCCESS(result);
    ASSERT_LLVM_ERROR(checkResult(test.outputs[0], **result));

    for (auto arg : inputArguments) {
      delete arg;
    }
  }
}

std::string printEndToEndDesc(const testing::TestParamInfo<EndToEndDesc> desc) {
  return desc.param.description;
}

// Macro to define and end to end TestSuite that run test thanks the
// LambdaSupport according a EndToEndDesc
#define INSTANTIATE_END_TO_END_COMPILE_AND_RUN(TestSuite, lambdaSupport)       \
  TEST_P(TestSuite, compile_and_run) {                                         \
    auto desc = GetParam();                                                    \
    compile_and_run(desc, lambdaSupport);                                      \
  }

#define INSTANTIATE_END_TO_END_TEST_SUITE_FROM_FILE(prefix, suite,             \
                                                    lambdasupport, path)       \
  namespace prefix##suite {                                                    \
    auto valuesVector = loadEndToEndDesc(path);                                \
    auto values = testing::ValuesIn<std::vector<EndToEndDesc>>(valuesVector);  \
    INSTANTIATE_TEST_SUITE_P(prefix, suite, values, printEndToEndDesc);        \
  }

#define INSTANTIATE_END_TO_END_TEST_SUITE_FROM_ALL_TEST_FILES(suite,           \
                                                              lambdasupport)   \
                                                                               \
  class suite : public testing::TestWithParam<EndToEndDesc> {};                \
  INSTANTIATE_END_TO_END_COMPILE_AND_RUN(suite, lambdasupport)                 \
  INSTANTIATE_END_TO_END_TEST_SUITE_FROM_FILE(                                 \
      FHE, suite, lambdasupport, "tests/fixture/end_to_end_fhe.yaml")          \
  INSTANTIATE_END_TO_END_TEST_SUITE_FROM_FILE(                                 \
      EncryptedTensor, suite, lambdasupport,                                   \
      "tests/fixture/end_to_end_encrypted_tensor.yaml")                        \
  INSTANTIATE_END_TO_END_TEST_SUITE_FROM_FILE(                                 \
      FHELinalg, suite, lambdasupport,                                         \
      "tests/fixture/end_to_end_fhelinalg.yaml")

/// Instantiate the test suite for Jit
INSTANTIATE_END_TO_END_TEST_SUITE_FROM_ALL_TEST_FILES(
    JitTest, mlir::concretelang::JITSupport())

/// Instantiate the test suite for Jit
INSTANTIATE_END_TO_END_TEST_SUITE_FROM_ALL_TEST_FILES(
    LibraryTest, mlir::concretelang::LibrarySupport("/tmp/end_to_end_test_" +
                                                    desc.description))
