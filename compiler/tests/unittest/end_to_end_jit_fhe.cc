
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

// FHE.apply_lookup_table /////////////////////////////////////////////////////

class ApplyLookupTableWithPrecision : public ::testing::TestWithParam<int> {};

TEST_P(ApplyLookupTableWithPrecision, identity_func) {
  uint64_t precision = GetParam();
  std::ostringstream mlirProgram;
  uint64_t sizeOfTLU = 1 << precision;

  mlirProgram << "func @main(%arg0: !FHE.eint<" << precision
              << ">) -> !FHE.eint<" << precision << "> { \n"
              << "    %tlu = arith.constant dense<[0";

  for (uint64_t i = 1; i < sizeOfTLU; i++)
    mlirProgram << ", " << i;

  mlirProgram << "]> : tensor<" << sizeOfTLU << "xi64>\n"
              << "    %1 = \"FHE.apply_lookup_table\"(%arg0, %tlu): "
              << "(!FHE.eint<" << precision << ">, tensor<" << sizeOfTLU
              << "xi64>) -> (!FHE.eint<" << precision << ">)\n "
              << "return %1: !FHE.eint<" << precision << ">\n"
              << "}\n";

  checkedJit(lambda, mlirProgram.str());

  if (precision >= 6) {
    // This test often fails for this precision, so we need retries.
    // Reason: the current encryption parameters are a little short for this
    // precision.

    static const int max_tries = 10;

    for (uint64_t i = 0; i < sizeOfTLU; i++) {
      for (int retry = 0; retry <= max_tries; retry++) {
        if (retry == max_tries)
          GTEST_FATAL_FAILURE_("Maximum number of tries exceeded");

        llvm::Expected<uint64_t> val = lambda(i);
        ASSERT_EXPECTED_SUCCESS(val);

        if (*val == i)
          break;
      }
    }
  } else {
    for (uint64_t i = 0; i < sizeOfTLU; i++)
      ASSERT_EXPECTED_VALUE(lambda(i), i);
  }
}