#include "end_to_end_fixture/EndToEndFixture.h"
#include <concretelang/Runtime/DFRuntime.hpp>
#define BENCHMARK_HAS_CXX11
#include "llvm/Support/Path.h"
#include <benchmark/benchmark.h>

#include "tests_tools/StackSize.h"
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
  std::vector<const mlir::concretelang::LambdaArgument *> inputArguments;
  inputArguments.reserve(test.inputs.size());
  for (auto input : test.inputs) {
    inputArguments.push_back(&input.getValue());
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
  std::vector<const mlir::concretelang::LambdaArgument *> inputArguments;
  inputArguments.reserve(test.inputs.size());
  for (auto input : test.inputs) {
    inputArguments.push_back(&input.getValue());
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

static int registerEndToEndTestFromFile(std::string prefix, std::string path,
                                        size_t stackSizeRequirement = 0) {
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
  setCurrentStackLimit(stackSizeRequirement);
  mlir::concretelang::CompilationOptions defaul;
  registe("default", defaul);
  mlir::concretelang::CompilationOptions loop;
  loop.loopParallelize = true;
  registe("loop", loop);
#ifdef CONCRETELANG_CUDA_SUPPORT
  mlir::concretelang::CompilationOptions gpu;
  gpu.useGPU = true;
  registe("gpu", gpu);
#endif
#ifdef CONCRETELANG_DATAFLOW_EXECUTION_ENABLED
  mlir::concretelang::CompilationOptions dataflow;
  dataflow.dataflowParallelize = true;
  dataflow.loopParallelize = true;
  registe("dataflow", dataflow);
#endif
  return 1;
}

int main(int argc, char **argv) {
  char *bench_name = "MLBench";
  char *file_name =
      "tests/end_to_end_benchmarks/mlbench/end_to_end_mlbench.yaml";
  size_t stack_size = 0;

  char *env = getenv("BENCHMARK_NAME");
  if (env != nullptr)
    bench_name = env;
  env = getenv("BENCHMARK_FILE");
  if (env != nullptr)
    file_name = env;
  env = getenv("BENCHMARK_STACK");
  if (env != nullptr)
    stack_size = strtoul(env, NULL, 10);

  std::cout << "Benchmark executing [" << bench_name << "] from file "
            << file_name << "\n";
  registerEndToEndTestFromFile(bench_name, file_name, stack_size);

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  _dfr_terminate();
  return 0;
}
