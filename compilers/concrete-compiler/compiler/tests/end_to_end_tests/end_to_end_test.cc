#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <regex>
#include <type_traits>

#include "concretelang/Common/Values.h"
#include "concretelang/Support/CompilationFeedback.h"
#include "concretelang/TestLib/TestProgram.h"
#include "end_to_end_fixture/EndToEndFixture.h"
#include "end_to_end_jit_test.h"
#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/assert.h"
#include "tests_tools/keySetCache.h"

using concretelang::testlib::TestProgram;
using concretelang::values::Value;

/// @brief EndToEndTest is a template that allows testing for one program for a
/// TestDescription.
class EndToEndTest : public ::testing::Test {
public:
  explicit EndToEndTest(std::string program, TestDescription desc,
                        std::optional<TestErrorRate> errorRate,
                        EndToEndTestOptions options)
      : program(program), desc(desc), errorRate(errorRate),
        testCircuit(std::nullopt), options(options) {
    if (errorRate.has_value()) {
      options.compilationOptions.optimizerConfig.global_p_error =
          errorRate->global_p_error;
      options.compilationOptions.optimizerConfig.p_error =
          errorRate->global_p_error;
    }
  };

  void SetUp() override {
    TestProgram tc(options.compilationOptions);
    ASSERT_OUTCOME_HAS_VALUE(tc.compile({program}));
    ASSERT_OUTCOME_HAS_VALUE(tc.generateKeyset());
    testCircuit.emplace(std::move(tc));
    args = std::vector<Value>();
    for (auto &input : desc.inputs) {
      args.push_back(input.getValue());
    }
  }

  void TearDown() override {}

  void TestBody() override {
    if (!errorRate.has_value()) {
      testOnce();
    } else {
      testErrorRate();
    }
  }

  void testOnce() {
    for (auto tests_rep = 0; tests_rep <= options.numberOfRetry; tests_rep++) {
      // We execute the circuit.
      auto maybeRes = testCircuit->call(args);
      ASSERT_OUTCOME_HAS_VALUE(maybeRes);
      auto result = maybeRes.value();

      /* Check results */
      bool allgood = true;
      for (size_t i = 0; i < desc.outputs.size(); i++) {
        auto maybeErr = checkResult(desc.outputs[i], result[i]);
        if (maybeErr) {
          allgood = false;
          llvm::errs() << "/!\\ WARNING RETRY TEST: " << maybeErr << "\n";
          llvm::consumeError(std::move(maybeErr));
          break;
        }
        ASSERT_LLVM_ERROR(std::move(maybeErr));
      }

      // If OK we return
      if (allgood) {
        return;
      }

      // Otherwise we reset the keyset
      llvm::errs() << "Regenerating keyset with seed: ";
      llvm::errs() << tests_rep + 1;
      llvm::errs() << "\n";
      ASSERT_OUTCOME_HAS_VALUE(
          testCircuit->generateKeyset(tests_rep + 1, tests_rep + 1));
    }

    // If all attempts fail, we return an error.
    ASSERT_TRUE(false) << "Test failed after multiple attempts";
  }

  void testErrorRate() {
    auto nbError = 0;
    for (size_t i = 0; i < errorRate->nb_repetition; i++) {
      // We execute the circuit.
      auto maybeRes = (*testCircuit).call(args);
      ASSERT_OUTCOME_HAS_VALUE(maybeRes);
      auto result = maybeRes.value();

      /* Check result */
      // For now we support just one result
      assert(desc.outputs.size() == 1);
      auto err = checkResult(desc.outputs[0], result[0]);
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
  std::string artifactFolder;
  TestDescription desc;
  std::optional<TestErrorRate> errorRate;
  std::optional<mlir::concretelang::CompilerEngine::Library> library;
  std::optional<TestProgram> testCircuit;
  EndToEndTestOptions options;
  std::vector<Value> args;
};

std::string getTestName(EndToEndDesc desc, EndToEndTestOptions options,
                        int testNum) {
  std::ostringstream os;
  os << getOptionsName(options.compilationOptions) << "." << desc.description
     << "." << testNum;
  return std::regex_replace(os.str(), std::regex("-"), "");
}

void registerEndToEnd(std::string suiteName, std::string testName,
                      std::string valueName, std::string program,
                      TestDescription test,
                      std::optional<TestErrorRate> errorRate,
                      EndToEndTestOptions options) {
  // TODO: Get file and line from yaml
  auto file = __FILE__;
  auto line = __LINE__;
  ::testing::RegisterTest(
      suiteName.c_str(), testName.c_str(), nullptr, valueName.c_str(), file,
      line, [=]() -> EndToEndTest * {
        return new EndToEndTest(program, test, errorRate, options);
      });
}

void registerEndToEnd(std::string suiteName, EndToEndDesc desc,
                      EndToEndTestOptions options) {
  if (desc.v0Constraint.has_value()) {
    options.compilationOptions.v0FHEConstraints = desc.v0Constraint;
  }
  options.compilationOptions.optimizerConfig.encoding = desc.encoding;
  if (desc.p_error.has_value()) {
    options.compilationOptions.optimizerConfig.p_error = *desc.p_error;
    options.compilationOptions.optimizerConfig.global_p_error = NAN;
  }
  auto i = 0;
  for (auto test : desc.tests) {
    auto valueName = std::to_string(i);
    auto testName = getTestName(desc, options, i);
    if (desc.test_error_rates.empty()) {
      registerEndToEnd(suiteName, testName, valueName, desc.program, test,
                       std::nullopt, options);
    } else {
      auto j = 0;
      for (auto rate : desc.test_error_rates) {
        auto rateName = testName + "_rate" + std::to_string(j);
        registerEndToEnd(suiteName, rateName, valueName, desc.program, test,
                         rate, options);
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
void registerEndToEndSuite(std::string suiteName,
                           std::vector<EndToEndDesc> descriptions,
                           EndToEndTestOptions options) {
  for (auto desc : descriptions) {
    registerEndToEnd(suiteName, desc, options);
  }
}

namespace path = llvm::sys::path;

int main(int argc, char **argv) {

  // Parse google test options, update argc and argv by removing gtest options
  ::testing::InitGoogleTest(&argc, argv);

  // parse end to end test compiler options

  auto cmdLine = parseEndToEndCommandLine(argc, argv);

  auto options = std::get<0>(cmdLine);
  auto descriptionFiles = std::get<1>(cmdLine);

  for (auto descFile : descriptionFiles) {
    auto suiteName = path::stem(descFile.path).str();
    registerEndToEndSuite(suiteName, descFile.descriptions, options);
  }
  return RUN_ALL_TESTS();
}
