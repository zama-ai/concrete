#include "../end_to_end_tests/end_to_end_test.h"

#include <benchmark/benchmark.h>

#define BENCHMARK_HAS_CXX11
#include "llvm/Support/Path.h"

#include "tests_tools/StackSize.h"
#include "tests_tools/keySetCache.h"

#define check(expr)                                                            \
  if (auto E = expr.takeError()) {                                             \
    std::cerr << "Error: " << llvm::toString(std::move(E)) << "\n";            \
    assert(false && "See error above");                                        \
  }

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
  check(compilationResult);

  auto clientParameters = support.loadClientParameters(**compilationResult);
  check(clientParameters);

  for (auto _ : state) {
    check(support.keySet(*clientParameters, llvm::None));
  }
}

/// Benchmark time of the encryption
template <typename LambdaSupport>
static void BM_ExportArguments(benchmark::State &state,
                               EndToEndDesc description, LambdaSupport support,
                               mlir::concretelang::CompilationOptions options) {
  auto compilationResult = support.compile(description.program, options);
  check(compilationResult);

  auto clientParameters = support.loadClientParameters(**compilationResult);
  check(clientParameters);

  auto keySet = support.keySet(*clientParameters, getTestKeySetCache());
  check(keySet);

  assert(description.tests.size() > 0);
  auto test = description.tests[0];
  std::vector<const mlir::concretelang::LambdaArgument *> inputArguments;
  inputArguments.reserve(test.inputs.size());
  for (auto input : test.inputs) {
    inputArguments.push_back(&input.getValue());
  }

  for (auto _ : state) {
    check(support.exportArguments(*clientParameters, **keySet, inputArguments));
  }
}

/// Benchmark time of the program evaluation
template <typename LambdaSupport>
static void BM_Evaluate(benchmark::State &state, EndToEndDesc description,
                        LambdaSupport support,
                        mlir::concretelang::CompilationOptions options) {
  options.optimizerConfig.display = true;
  auto compilationResult = support.compile(description.program, options);
  check(compilationResult);
  auto clientParameters = support.loadClientParameters(**compilationResult);
  check(clientParameters);
  auto keySet = support.keySet(*clientParameters, getTestKeySetCache());
  check(keySet);
  assert(description.tests.size() > 0);
  auto test = description.tests[0];
  std::vector<const mlir::concretelang::LambdaArgument *> inputArguments;
  inputArguments.reserve(test.inputs.size());
  for (auto input : test.inputs) {
    inputArguments.push_back(&input.getValue());
  }
  auto publicArguments =
      support.exportArguments(*clientParameters, **keySet, inputArguments);
  check(publicArguments);

  auto serverLambda = support.loadServerLambda(**compilationResult);
  check(serverLambda);
  auto evaluationKeys = (*keySet)->evaluationKeys();

  // Warmup
  assert(support.serverCall(*serverLambda, **publicArguments, evaluationKeys));

  for (auto _ : state) {
    check(support.serverCall(*serverLambda, **publicArguments, evaluationKeys));
  }
}

std::string getOptionsName(mlir::concretelang::CompilationOptions options) {
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
  return os.str().substr(1);
}

enum Action {
  COMPILE,
  KEYGEN,
  ENCRYPT,
  EVALUATE,
};

void registerEndToEndBenchmark(std::string suiteName,
                               std::vector<EndToEndDesc> descriptions,
                               mlir::concretelang::CompilationOptions options,
                               std::vector<enum Action> actions,
                               size_t stackSizeRequirement = 0) {
  auto optionsName = getOptionsName(options);
  for (auto description : descriptions) {
    options.clientParametersFuncName = "main";
    mlir::concretelang::JITSupport support;
    auto benchName = [&](std::string name) {
      std::ostringstream s;
      s << suiteName << "/" << name << "/" << optionsName << "/"
        << description.description;
      return s.str();
    };
    for (auto action : actions) {
      switch (action) {
      case Action::COMPILE:
        benchmark::RegisterBenchmark(
            benchName("compile").c_str(), [=](::benchmark::State &st) {
              BM_Compile(st, description, support, options);
            });
        break;
      case Action::KEYGEN:
        benchmark::RegisterBenchmark(
            benchName("keygen").c_str(), [=](::benchmark::State &st) {
              BM_KeyGen(st, description, support, options);
            });
        break;
      case Action::ENCRYPT:
        benchmark::RegisterBenchmark(
            benchName("encrypt").c_str(), [=](::benchmark::State &st) {
              BM_ExportArguments(st, description, support, options);
            });
        break;
      case Action::EVALUATE:
        benchmark::RegisterBenchmark(
            benchName("evaluate").c_str(), [=](::benchmark::State &st) {
              BM_Evaluate(st, description, support, options);
            });
        break;
      }
    }
  }
  setCurrentStackLimit(stackSizeRequirement);
}

int main(int argc, char **argv) {
  // Parse google benchmark options
  ::benchmark::Initialize(&argc, argv);

  llvm::cl::list<enum Action> clActions(
      "b", "bench",
      llvm::cl::desc("Specify benchmark cases to run, if no benchmarks speci"),
      llvm::cl::values(
          clEnumValN(Action::COMPILE, "compile", "Run compile benchmark")),
      llvm::cl::values(
          clEnumValN(Action::KEYGEN, "keygen", "Run keygen benchmark")),
      llvm::cl::values(
          clEnumValN(Action::ENCRYPT, "encrypt", "Run encrypt benchmark")),
      llvm::cl::values(
          clEnumValN(Action::EVALUATE, "evaluate", "Run evaluate benchmark")));

  // parse end to end test compiler options
  auto options = parseEndToEndCommandLine(argc, argv);

  auto compilationOptions = std::get<0>(options);
  auto libpath = std::get<1>(options);
  auto descriptionFiles = std::get<2>(options);

  std::vector<enum Action> actions = clActions;
  if (actions.empty()) {
    actions = {Action::COMPILE, Action::KEYGEN, Action::ENCRYPT,
               Action::EVALUATE};
  }

  auto stackSizeRequirement = 0;
  for (auto descFile : descriptionFiles) {
    auto suiteName = llvm::sys::path::stem(descFile.path).str();
    registerEndToEndBenchmark(suiteName, descFile.descriptions,
                              compilationOptions, actions,
                              stackSizeRequirement);
  }
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
