#include "end_to_end_fixture/EndToEndFixture.h"
#define BENCHMARK_HAS_CXX11
#include "llvm/Support/Path.h"
#include <benchmark/benchmark.h>

#include "tests_tools/keySetCache.h"

/// Benchmark time of the compilation
template <typename LambdaSupport>
static void BM_Compile(benchmark::State &state, EndToEndDesc description,
                       LambdaSupport support,
                       mlir::concretelang::CompilationOptions options) {
  for (auto _ : state) {
    if (support.compile(description.program, options)) {
    };
  }
}

/// Benchmark time of the key generation
template <typename LambdaSupport>
static void BM_KeyGen(benchmark::State &state, EndToEndDesc description,
                      LambdaSupport support,
                      mlir::concretelang::CompilationOptions options) {
  auto compilationResult = support.compile(description.program, options);
  assert(compilationResult);

  auto clientParameters = support.loadClientParameters(**compilationResult);
  assert(clientParameters);

  for (auto _ : state) {
    assert(support.keySet(*clientParameters, llvm::None));
  }
}

/// Benchmark time of the encryption
template <typename LambdaSupport>
static void BM_ExportArguments(benchmark::State &state,
                               EndToEndDesc description, LambdaSupport support,
                               mlir::concretelang::CompilationOptions options) {
  auto compilationResult = support.compile(description.program, options);
  assert(compilationResult);

  auto clientParameters = support.loadClientParameters(**compilationResult);
  assert(clientParameters);

  auto keySet = support.keySet(*clientParameters, getTestKeySetCache());
  assert(keySet);

  assert(description.tests.size() > 0);
  auto test = description.tests[0];
  std::vector<mlir::concretelang::LambdaArgument *> inputArguments;
  inputArguments.reserve(test.inputs.size());
  for (auto input : test.inputs) {
    auto arg = valueDescriptionToLambdaArgument(input);
    assert(arg);
    inputArguments.push_back(arg.get());
  }

  for (auto _ : state) {
    assert(
        support.exportArguments(*clientParameters, **keySet, inputArguments));
  }
}

/// Benchmark time of the program evaluation
template <typename LambdaSupport>
static void BM_Evaluate(benchmark::State &state, EndToEndDesc description,
                        LambdaSupport support,
                        mlir::concretelang::CompilationOptions options) {
  auto compilationResult = support.compile(description.program, options);
  assert(compilationResult);

  auto clientParameters = support.loadClientParameters(**compilationResult);
  assert(clientParameters);

  auto keySet = support.keySet(*clientParameters, getTestKeySetCache());
  assert(keySet);

  assert(description.tests.size() > 0);
  auto test = description.tests[0];
  std::vector<mlir::concretelang::LambdaArgument *> inputArguments;
  inputArguments.reserve(test.inputs.size());
  for (auto input : test.inputs) {
    auto arg = valueDescriptionToLambdaArgument(input);
    assert(arg);
    inputArguments.push_back(arg.get());
  }

  auto publicArguments =
      support.exportArguments(*clientParameters, **keySet, inputArguments);
  assert(publicArguments);

  auto serverLambda = support.loadServerLambda(**compilationResult);
  assert(serverLambda);
  auto evaluationKeys = (*keySet)->evaluationKeys();

  for (auto _ : state) {
    assert(
        support.serverCall(*serverLambda, **publicArguments, evaluationKeys));
  }
}

static int registerEndToEndTestFromFile(std::string prefix, std::string path) {
  auto registe = [&](std::string optionsName,
                     mlir::concretelang::CompilationOptions options) {
    llvm::for_each(loadEndToEndDesc(path), [&](EndToEndDesc &description) {
      options.clientParametersFuncName = "main";
      mlir::concretelang::JITSupport support;
      auto benchName = [&](std::string name) {
        std::ostringstream s;
        s << prefix << "/" << name << "/" << optionsName << "/"
          << description.description;
        return s.str();
      };
      benchmark::RegisterBenchmark(
          benchName("Compile").c_str(), [=](::benchmark::State &st) {
            BM_Compile(st, description, support, options);
          });
      benchmark::RegisterBenchmark(
          benchName("KeyGen").c_str(), [=](::benchmark::State &st) {
            BM_KeyGen(st, description, support, options);
          });
      benchmark::RegisterBenchmark(
          benchName("ExportArguments").c_str(), [=](::benchmark::State &st) {
            BM_ExportArguments(st, description, support, options);
          });
      benchmark::RegisterBenchmark(
          benchName("Evaluate").c_str(), [=](::benchmark::State &st) {
            BM_Evaluate(st, description, support, options);
          });
      return;
    });
  };
  mlir::concretelang::CompilationOptions defaul;
  registe("default", defaul);
  mlir::concretelang::CompilationOptions loop;
  loop.loopParallelize = true;
  registe("loop", loop);
  // mlir::concretelang::CompilationOptions dataflow;
  // dataflow.dataflowParallelize = true;
  // registe("dataflow", dataflow);

  return 1;
}

auto _ = {
    registerEndToEndTestFromFile(
        "FHE", "tests/end_to_end_fixture/end_to_end_fhe.yaml"),
    registerEndToEndTestFromFile(
        "EncryptedTensor",
        "tests/end_to_end_fixture/end_to_end_encrypted_tensor.yaml"),
    registerEndToEndTestFromFile(
        "FHELinalg", "tests/end_to_end_fixture/end_to_end_fhelinalg.yaml"),
    registerEndToEndTestFromFile(
        "FHELinalg", "tests/end_to_end_fixture/end_to_end_programs.yaml")};

BENCHMARK_MAIN();
