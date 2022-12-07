#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <type_traits>

#include "concretelang/Support/CompilationFeedback.h"
#include "concretelang/Support/JITSupport.h"
#include "concretelang/Support/LibrarySupport.h"
#include "end_to_end_fixture/EndToEndFixture.h"
#include "end_to_end_jit_test.h"
#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/keySetCache.h"

/// @brief EndToEndTest is a template that allows testing for one program for a
/// TestDescription using a LambdaSupport.
template <typename LambdaSupport> class EndToEndTest : public ::testing::Test {
public:
  explicit EndToEndTest(std::string program, TestDescription desc,
                        llvm::Optional<TestErrorRate> errorRate,
                        LambdaSupport support,
                        mlir::concretelang::CompilationOptions options)
      : program(program), desc(desc), errorRate(errorRate), support(support),
        options(options) {
    if (errorRate.hasValue()) {
      options.optimizerConfig.global_p_error = errorRate->global_p_error;
      options.optimizerConfig.p_error = errorRate->global_p_error;
    }
  };

  void SetUp() override {
    /* Compile the program */
    auto expectCompilationResult = support.compile(program, options);
    ASSERT_EXPECTED_SUCCESS(expectCompilationResult);

    /* Load the client parameters */
    auto expectClientParameters =
        support.loadClientParameters(**expectCompilationResult);
    ASSERT_EXPECTED_SUCCESS(expectClientParameters);
    clientParameters = *expectClientParameters;

    /* Build the keyset */
    auto expectKeySet = support.keySet(clientParameters, getTestKeySetCache());
    ASSERT_EXPECTED_SUCCESS(expectKeySet);
    keySet = std::move(*expectKeySet);

    /* Load the server lambda */
    auto expectServerLambda =
        support.loadServerLambda(**expectCompilationResult);
    ASSERT_EXPECTED_SUCCESS(expectServerLambda);
    serverLambda = *expectServerLambda;

    /* Create the public argument */
    std::vector<const mlir::concretelang::LambdaArgument *> inputArguments;
    inputArguments.reserve(desc.inputs.size());

    for (auto &input : desc.inputs) {
      inputArguments.push_back(&input.getValue());
    }
    auto expectPublicArguments =
        support.exportArguments(clientParameters, *keySet, inputArguments);
    ASSERT_EXPECTED_SUCCESS(expectPublicArguments);
    publicArguments = std::move(*expectPublicArguments);
  }

  void TestBody() override {
    if (!errorRate.hasValue()) {
      testOnce();
    } else {
      testErrorRate();
    }
  }

  void testOnce() {
    auto evaluationKeys = keySet->evaluationKeys();
    /* Call the server lambda */
    auto publicResult =
        support.serverCall(serverLambda, *publicArguments, evaluationKeys);
    ASSERT_EXPECTED_SUCCESS(publicResult);

    /* Decrypt the public result */
    auto result = mlir::concretelang::typedResult<
        std::unique_ptr<mlir::concretelang::LambdaArgument>>(*keySet,
                                                             **publicResult);
    ASSERT_EXPECTED_SUCCESS(result);

    /* Check result */
    // For now we support just one result
    assert(desc.outputs.size() == 1);
    ASSERT_LLVM_ERROR(checkResult(desc.outputs[0], **result));
  }

  void testErrorRate() {
    auto evaluationKeys = keySet->evaluationKeys();
    auto nbError = 0;
    for (size_t i = 0; i < errorRate->nb_repetition; i++) {
      /* Call the server lambda */
      auto publicResult =
          support.serverCall(serverLambda, *publicArguments, evaluationKeys);
      ASSERT_EXPECTED_SUCCESS(publicResult);

      /* Decrypt the public result */
      auto result = mlir::concretelang::typedResult<
          std::unique_ptr<mlir::concretelang::LambdaArgument>>(*keySet,
                                                               **publicResult);
      ASSERT_EXPECTED_SUCCESS(result);

      /* Check result */
      // For now we support just one result
      assert(desc.outputs.size() == 1);
      auto err = checkResult(desc.outputs[0], **result);
      if (err) {
        nbError++;
        DISCARD_LLVM_ERROR(err);
      }
    }
    double threshold = errorRate->too_high_error_count_threshold();
    std::cout << "n_rep " << errorRate->nb_repetition << " p_error "
              << errorRate->global_p_error << " maximum_errors " << threshold
              << "\n";
    ASSERT_LE(nbError, threshold) << "Empirical error rate is too high";
  }

private:
  std::string program;
  TestDescription desc;
  llvm::Optional<TestErrorRate> errorRate;
  LambdaSupport support;
  mlir::concretelang::CompilationOptions options;

  // Initialized by the SetUp
  typename LambdaSupport::lambda serverLambda;
  mlir::concretelang::ClientParameters clientParameters;
  std::unique_ptr<concretelang::clientlib::KeySet> keySet;
  std::unique_ptr<concretelang::clientlib::PublicArguments> publicArguments;
};

std::string getTestName(EndToEndDesc desc,
                        mlir::concretelang::CompilationOptions options,
                        int testNum) {
  std::ostringstream os;
  if (options.loopParallelize)
    os << "_loop";
  if (options.dataflowParallelize)
    os << "_dataflow";
  if (options.emitGPUOps)
    os << "_gpu";
  auto ostr = os.str();
  if (ostr.size() == 0) {
    os << "_default";
  }
  os << "." << desc.description << "." << testNum;
  return os.str().substr(1);
}

void registerEndToEnd(std::string suiteName, std::string testName,
                      std::string valueName, std::string libpath,
                      std::string program, TestDescription test,
                      llvm::Optional<TestErrorRate> errorRate,
                      mlir::concretelang::CompilationOptions options) {
  // TODO: Get file and line from yaml
  auto file = __FILE__;
  auto line = __LINE__;
  if (libpath.empty()) {
    ::testing::RegisterTest(
        suiteName.c_str(), testName.c_str(), nullptr, valueName.c_str(), file,
        line, [=]() -> EndToEndTest<mlir::concretelang::JITSupport> * {
          return new EndToEndTest<mlir::concretelang::JITSupport>(
              program, test, errorRate, mlir::concretelang::JITSupport(),
              options);
        });
  } else {
    ::testing::RegisterTest(
        suiteName.c_str(), testName.c_str(), nullptr, valueName.c_str(), file,
        line, [=]() -> EndToEndTest<mlir::concretelang::LibrarySupport> * {
          return new EndToEndTest<mlir::concretelang::LibrarySupport>(
              program, test, errorRate,
              mlir::concretelang::LibrarySupport(libpath), options);
        });
  }
}

void registerEndToEnd(std::string suiteName, std::string libpath,
                      EndToEndDesc desc,
                      mlir::concretelang::CompilationOptions options) {
  if (desc.v0Constraint.hasValue()) {
    options.v0FHEConstraints = desc.v0Constraint;
  }
  auto i = 0;
  for (auto test : desc.tests) {
    auto valueName = std::to_string(i);
    auto testName = getTestName(desc, options, i);
    if (desc.test_error_rates.empty()) {
      registerEndToEnd(suiteName, testName, valueName,
                       libpath.empty() ? libpath : libpath + desc.description,
                       desc.program, test, llvm::None, options);
    } else {
      auto j = 0;
      for (auto rate : desc.test_error_rates) {
        auto rateName = testName + "_rate" + std::to_string(j);
        registerEndToEnd(suiteName, rateName, valueName,
                         libpath.empty() ? libpath : libpath + desc.description,
                         desc.program, test, rate, options);
        j++;
      }
    }
    i++;
  }
}

/// @brief Register a suite of end to end test
/// @param suiteName The name of the suite.
/// @param descriptions A vector of description of tests to register .
/// @param options The compilation options.
void registerEndToEndSuite(std::string suiteName, std::string libpath,
                           std::vector<EndToEndDesc> descriptions,
                           mlir::concretelang::CompilationOptions options) {
  for (auto desc : descriptions) {
    registerEndToEnd(suiteName, libpath, desc, options);
  }
}

namespace path = llvm::sys::path;

int main(int argc, char **argv) {

  // Parse google test options, update argc and argv by removing gtest options
  ::testing::InitGoogleTest(&argc, argv);

  // parse end to end test compiler options

  auto options = parseEndToEndCommandLine(argc, argv);

  auto compilationOptions = std::get<0>(options);
  auto libpath = std::get<1>(options);
  auto descriptionFiles = std::get<2>(options);

  for (auto descFile : descriptionFiles) {
    auto suiteName = path::stem(descFile.path).str();
    if (libpath.empty()) {
      suiteName = suiteName + ".jit";
    } else {
      suiteName = suiteName + ".library";
    }
    registerEndToEndSuite(suiteName, libpath, descFile.descriptions,
                          compilationOptions);
  }
  return RUN_ALL_TESTS();
}
