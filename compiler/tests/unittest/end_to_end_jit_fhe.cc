
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "EndToEndFixture.h"

class EndToEndJitTest : public testing::TestWithParam<EndToEndDesc> {};

TEST_P(EndToEndJitTest, compile_and_run) {
  EndToEndDesc desc = GetParam();

  // Compile program
  // mlir::concretelang::JitCompilerEngine::Lambda lambda =
  checkedJit(lambda, desc.program);

  // Prepare arguments
  for (auto test : desc.tests) {
    std::vector<mlir::concretelang::LambdaArgument *> inputArguments;
    inputArguments.reserve(test.inputs.size());
    for (auto input : test.inputs) {
      auto arg = valueDescriptionToLambdaArgument(input);
      ASSERT_EXPECTED_SUCCESS(arg);
      inputArguments.push_back(arg.get());
    }

    // Call the lambda
    auto res =
        lambda.operator()<std::unique_ptr<mlir::concretelang::LambdaArgument>>(
            llvm::ArrayRef<mlir::concretelang::LambdaArgument *>(
                inputArguments));
    ASSERT_EXPECTED_SUCCESS(res);
    if (test.outputs.size() != 1) {
      FAIL() << "Only one result function are supported.";
    }
    ASSERT_LLVM_ERROR(checkResult(test.outputs[0], res.get()));

    // Free arguments
    for (auto arg : inputArguments) {
      delete arg;
    }
  }
}

#define INSTANTIATE_END_TO_END_JIT_TEST_SUITE_FROM_FILE(prefix, path)          \
  namespace prefix {                                                           \
  auto valuesVector = loadEndToEndDesc(path);                                  \
  auto values = testing::ValuesIn<std::vector<EndToEndDesc>>(valuesVector);    \
  INSTANTIATE_TEST_SUITE_P(prefix, EndToEndJitTest, values,                    \
                           printEndToEndDesc);                                 \
  }

INSTANTIATE_END_TO_END_JIT_TEST_SUITE_FROM_FILE(
    FHE, "tests/unittest/end_to_end_fhe.yaml")
INSTANTIATE_END_TO_END_JIT_TEST_SUITE_FROM_FILE(
    EncryptedTensor, "tests/unittest/end_to_end_encrypted_tensor.yaml")
