#include "concretelang/Common/Compat.h"
#include "concretelang/TestLib/TestProgram.h"
#include "end_to_end_fixture/EndToEndFixture.h"
#include <concretelang/Runtime/DFRuntime.hpp>
#define BENCHMARK_HAS_CXX11
#include "tests_tools/StackSize.h"
#include "tests_tools/keySetCache.h"
#include "llvm/Support/Path.h"
#include <benchmark/benchmark.h>

using namespace concretelang::testlib;

/// Benchmark time of the compilation
static void BM_Compile(benchmark::State &state, EndToEndDesc description,
                       mlir::concretelang::CompilerEngine engine,
                       mlir::concretelang::CompilationOptions options) {
  engine.setCompilationOptions(options);
  std::vector<std::string> sources = {description.program};
  auto artifactFolder = createTempFolderIn(getSystemTempFolderPath());
  for (auto _ : state) {
    if (engine.compile(sources, artifactFolder)) {
    };
  }
}

/// Benchmark time of the key generation
static void BM_KeyGen(benchmark::State &state, EndToEndDesc description,
                      mlir::concretelang::CompilerEngine engine,
                      mlir::concretelang::CompilationOptions options) {
  engine.setCompilationOptions(options);
  std::vector<std::string> sources = {description.program};

  auto artifactFolder = createTempFolderIn(getSystemTempFolderPath());
  auto result = engine.compile(sources, artifactFolder);
  assert(result);

  for (auto _ : state) {
    assert(getTestKeySetCachePtr()->getKeyset(
        result->getProgramInfo().asReader().getKeyset(), 0, 0, 0, 0));
  }
}

/// Benchmark time of the encryption
static void BM_ExportArguments(benchmark::State &state,
                               EndToEndDesc description,
                               mlir::concretelang::CompilerEngine engine,
                               mlir::concretelang::CompilationOptions options) {
  engine.setCompilationOptions(options);
  std::vector<std::string> sources = {description.program};

  auto artifactFolder = createTempFolderIn(getSystemTempFolderPath());
  auto compiled = engine.compile(sources, artifactFolder);
  assert(compiled);
  auto programInfo = compiled->getProgramInfo();
  auto keyset = getTestKeySetCachePtr()
                    ->getKeyset(programInfo.asReader().getKeyset(), 0, 0, 0, 0)
                    .value();
  auto csprng = std::make_shared<::concretelang::csprng::EncryptionCSPRNG>(
      ::concretelang::csprng::EncryptionCSPRNG(0));

  auto circuit = ClientCircuit::create(programInfo.asReader().getCircuits()[0],
                                       keyset.client, csprng, false)
                     .value();

  assert(description.tests.size() > 0);
  auto test = description.tests[0];
  auto inputArguments = std::vector<TransportValue>();
  inputArguments.reserve(test.inputs.size());

  for (auto _ : state) {
    for (size_t i = 0; i < test.inputs.size(); i++) {
      auto input = circuit.prepareInput(test.inputs[i].getValue(), i).value();
      inputArguments.push_back(input);
    }
  }
}

/// Benchmark time of the program evaluation
static void BM_Evaluate(benchmark::State &state, EndToEndDesc description,
                        mlir::concretelang::CompilerEngine engine,
                        mlir::concretelang::CompilationOptions options) {
  engine.setCompilationOptions(options);
  std::vector<std::string> sources = {description.program};

  auto artifactFolder = createTempFolderIn(getSystemTempFolderPath());
  auto compiled = engine.compile(sources, artifactFolder);
  assert(compiled);
  auto programInfo = compiled->getProgramInfo();
  auto keyset = getTestKeySetCachePtr()
                    ->getKeyset(programInfo.asReader().getKeyset(), 0, 0, 0, 0)
                    .value();
  auto csprng = std::make_shared<::concretelang::csprng::EncryptionCSPRNG>(
      ::concretelang::csprng::EncryptionCSPRNG(0));
  auto clientCircuit =
      ClientCircuit::create(programInfo.asReader().getCircuits()[0],
                            keyset.client, csprng, false)
          .value();

  assert(description.tests.size() > 0);
  auto test = description.tests[0];
  auto inputArguments = std::vector<TransportValue>();
  inputArguments.reserve(test.inputs.size());

  for (size_t i = 0; i < test.inputs.size(); i++) {
    auto input =
        clientCircuit.prepareInput(test.inputs[i].getValue(), i).value();
    inputArguments.push_back(input);
  }

  auto serverProgram = ServerProgram::load(
      programInfo, compiled->getSharedLibraryPath(compiled->getOutputDirPath()),
      false);
  auto serverCircuit =
      serverProgram.value()
          .getServerCircuit(programInfo.asReader().getCircuits()[0].getName())
          .value();

  // Warmup
  assert(serverCircuit.call(keyset.server, inputArguments));

  for (auto _ : state) {
    assert(serverCircuit.call(keyset.server, inputArguments));
  }
}

static int registerEndToEndTestFromFile(std::string prefix, std::string path,
                                        size_t stackSizeRequirement = 0) {
  auto registe = [&](std::string optionsName,
                     mlir::concretelang::CompilationOptions options) {
    llvm::for_each(loadEndToEndDesc(path), [&](EndToEndDesc &description) {
      options.mainFuncName = "main";
      auto context = mlir::concretelang::CompilationContext::createShared();
      mlir::concretelang::CompilerEngine engine(context);
      auto benchName = [&](std::string name) {
        std::ostringstream s;
        s << prefix << "/" << name << "/" << optionsName << "/"
          << description.description;
        return s.str();
      };
      benchmark::RegisterBenchmark(
          benchName("Compile").c_str(), [=](::benchmark::State &st) {
            BM_Compile(st, description, engine, options);
          });
      benchmark::RegisterBenchmark(
          benchName("KeyGen").c_str(), [=](::benchmark::State &st) {
            BM_KeyGen(st, description, engine, options);
          });
      benchmark::RegisterBenchmark(
          benchName("ExportArguments").c_str(), [=](::benchmark::State &st) {
            BM_ExportArguments(st, description, engine, options);
          });
      benchmark::RegisterBenchmark(
          benchName("Evaluate").c_str(), [=](::benchmark::State &st) {
            BM_Evaluate(st, description, engine, options);
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
  gpu.emitGPUOps = true;
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
  char const *bench_name = "MLBench";
  char const *file_name =
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
