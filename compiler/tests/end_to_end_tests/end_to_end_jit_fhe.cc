
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "concretelang/Support/CompilationFeedback.h"
#include "concretelang/Support/JITSupport.h"
#include "concretelang/Support/LibrarySupport.h"
#include "end_to_end_fixture/EndToEndFixture.h"
#include "end_to_end_jit_test.h"
#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/keySetCache.h"

#define CHECK_OR_ERROR(val)                                                    \
  {                                                                            \
    if (!bool(val)) {                                                          \
      return StreamStringError(llvm::toString(std::move(val.takeError())) +    \
                               "\nInvalid '" #val "'");                        \
    }                                                                          \
  }

using mlir::concretelang::StreamStringError;

template <typename LambdaSupport>
void compile_and_run(EndToEndDesc desc, LambdaSupport support) {
  mlir::concretelang::CompilationOptions options("main");
  options.loopParallelize = desc.loopParallelize;
  options.dataflowParallelize = desc.dataflowParallelize;
  options.asyncOffload = desc.asyncOffload;
  if (desc.v0Constraint.hasValue()) {
    options.v0FHEConstraints = *desc.v0Constraint;
  }
  if (desc.v0Parameter.hasValue()) {
    options.v0Parameter = *desc.v0Parameter;
  }
  if (desc.largeIntegerParameter.hasValue()) {
    options.largeIntegerParameter = *desc.largeIntegerParameter;
  }
  if (desc.test_error_rates.empty()) {
    compile_and_run_for_config(desc, support, options, llvm::None);
  } else {
    for (auto test_error_rate : desc.test_error_rates) {
      options.optimizerConfig.p_error = test_error_rate.p_error;
      compile_and_run_for_config(desc, support, options, test_error_rate);
    }
  }
}

template <typename LambdaSupport>
void compile_and_run_for_config(EndToEndDesc desc, LambdaSupport support,
                                mlir::concretelang::CompilationOptions options,
                                llvm::Optional<TestErrorRate> test_error_rate) {
  /* 1 - Compile the program */
  auto compilationResult = support.compile(desc.program, options);
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

  // Just test that we can load the compilation feedback
  auto feedback = support.loadCompilationFeedback(**compilationResult);
  ASSERT_EXPECTED_SUCCESS(feedback);

  assert_all_test_entries(desc, test_error_rate, support, keySet,
                          evaluationKeys, clientParameters, serverLambda);
}

template <typename LambdaSupport, typename KeySet, typename EvaluationKeys,
          typename ClientParameters, typename ServerLambda>
llvm::Error run_once_1_test_entry_once(TestDescription &test,
                                       LambdaSupport &support, KeySet &keySet,
                                       EvaluationKeys &evaluationKeys,
                                       ClientParameters &clientParameters,
                                       ServerLambda &serverLambda) {
  std::vector<mlir::concretelang::LambdaArgument *> inputArguments;
  inputArguments.reserve(test.inputs.size());
  for (auto input : test.inputs) {
    auto arg = valueDescriptionToLambdaArgument(input);
    CHECK_OR_ERROR(arg);
    inputArguments.push_back(arg.get());
  }

  /* 4 - Create the public arguments */
  auto publicArguments =
      support.exportArguments(*clientParameters, **keySet, inputArguments);
  CHECK_OR_ERROR(publicArguments);

  /* 5 - Call the server lambda */
  auto publicResult =
      support.serverCall(*serverLambda, **publicArguments, evaluationKeys);
  CHECK_OR_ERROR(publicResult);

  /* 6 - Decrypt the public result */
  auto result = mlir::concretelang::typedResult<
      std::unique_ptr<mlir::concretelang::LambdaArgument>>(**keySet,
                                                           **publicResult);

  /* 7 - Check result */
  CHECK_OR_ERROR(result);
  auto error = checkResult(test.outputs[0], **result);
  for (auto arg : inputArguments) {
    delete arg;
  }
  return error;
}

template <typename LambdaSupport, typename KeySet, typename EvaluationKeys,
          typename ClientParameters, typename ServerLambda>
void assert_all_test_entries(EndToEndDesc &desc,
                             llvm::Optional<TestErrorRate> &opt_test_error_rate,
                             LambdaSupport &support, KeySet &keySet,
                             EvaluationKeys &evaluationKeys,
                             ClientParameters &clientParameters,
                             ServerLambda &serverLambda) {
  auto run = [&](TestDescription &test) {
    return run_once_1_test_entry_once(test, support, keySet, evaluationKeys,
                                      clientParameters, serverLambda);
  };
  if (!opt_test_error_rate.has_value()) {
    for (auto test : desc.tests) {
      ASSERT_LLVM_ERROR(run(test));
    }
    return;
  }
  auto test_error_rate = opt_test_error_rate.value();
  ASSERT_LE(desc.tests.size(), test_error_rate.nb_repetition);
  int nb_error = 0;
  for (size_t i = 0; i < test_error_rate.nb_repetition; i++) {
    auto test = desc.tests[i % desc.tests.size()];
    auto error = run(test);
    if (error) {
      nb_error += 1;
      DISCARD_LLVM_ERROR(error);
    }
  }
  double maximum_errors = test_error_rate.too_high_error_count_threshold();
  // std::cout << "n_rep " << maximum_errors << " p_error " <<
  // test_error_rate.p_error <<  " maximum_errors " << maximum_errors << "\n";
  ASSERT_LE(nb_error, maximum_errors) << "Empirical error rate is too high";
}

std::string printEndToEndDesc(const testing::TestParamInfo<EndToEndDesc> desc) {
  return desc.param.description;
}

std::vector<EndToEndDesc> generateCustomVersions(std::string path) {
  std::vector<EndToEndDesc> cvdesc;
  auto add_custom = [&](EndToEndDesc d, std::string version,
                        bool loopParallelize, bool dataflowParallelize,
                        bool asyncOffload) {
    d.description = version + "__" + d.description;
    d.loopParallelize = loopParallelize;
    d.dataflowParallelize = dataflowParallelize;
    d.asyncOffload = asyncOffload;
    cvdesc.push_back(d);
    return;
  };
  for (auto d : loadEndToEndDesc(path)) {
    add_custom(d, "default", false, false, false);
    add_custom(d, "loop", true, false, false);
    add_custom(d, "async", false, false, true);
#ifdef CONCRETELANG_DATAFLOW_EXECUTION_ENABLED
    add_custom(d, "dataflow", false, true, false);
    add_custom(d, "dataflow_loop", true, true, false);
    add_custom(d, "dataflow_loop_async", true, true, true);
#endif
  }
  return cvdesc;
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
    auto valuesVector = generateCustomVersions(path);                          \
    auto values = testing::ValuesIn<std::vector<EndToEndDesc>>(valuesVector);  \
    INSTANTIATE_TEST_SUITE_P(prefix, suite, values, printEndToEndDesc);        \
  }

#define INSTANTIATE_END_TO_END_TEST_SUITE_FROM_ALL_TEST_FILES(suite,           \
                                                              lambdasupport)   \
                                                                               \
  class suite : public testing::TestWithParam<EndToEndDesc> {};                \
  INSTANTIATE_END_TO_END_COMPILE_AND_RUN(suite, lambdasupport)                 \
  INSTANTIATE_END_TO_END_TEST_SUITE_FROM_FILE(                                 \
      FHE, suite, lambdasupport,                                               \
      "tests/end_to_end_fixture/end_to_end_fhe.yaml")                          \
  INSTANTIATE_END_TO_END_TEST_SUITE_FROM_FILE(                                 \
      EncryptedTensor, suite, lambdasupport,                                   \
      "tests/end_to_end_fixture/end_to_end_encrypted_tensor.yaml")             \
  INSTANTIATE_END_TO_END_TEST_SUITE_FROM_FILE(                                 \
      FHELinalg, suite, lambdasupport,                                         \
      "tests/end_to_end_fixture/end_to_end_fhelinalg.yaml")                    \
  INSTANTIATE_END_TO_END_TEST_SUITE_FROM_FILE(                                 \
      FHELeveledOps, suite, lambdasupport,                                     \
      "tests/end_to_end_fixture/end_to_end_leveled.yaml")

/// Instantiate the test suite for Jit
INSTANTIATE_END_TO_END_TEST_SUITE_FROM_ALL_TEST_FILES(
    JitTest, mlir::concretelang::JITSupport())

/// Instantiate the test suite for Jit
INSTANTIATE_END_TO_END_TEST_SUITE_FROM_ALL_TEST_FILES(
    LibraryTest, mlir::concretelang::LibrarySupport("/tmp/end_to_end_test_" +
                                                    desc.description))
