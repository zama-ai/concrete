#include "../end_to_end_tests/end_to_end_test.h"
#include "concretelang/Common/Compat.h"
#include "concretelang/TestLib/TestCircuit.h"

#include <benchmark/benchmark.h>
#include <filesystem>

#define BENCHMARK_HAS_CXX11
#include "llvm/Support/Path.h"

#include "tests_tools/StackSize.h"
#include "tests_tools/keySetCache.h"

using namespace concretelang::testlib;

#define check(expr)                                                            \
  if (auto E = expr.takeError()) {                                             \
    std::cerr << "Error: " << llvm::toString(std::move(E)) << "\n";            \
    assert(false && "See error above");                                        \
  }

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
        result->getProgramInfo().asReader().getKeyset(), 0, 0));
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
  auto compiled = engine.compile(sources, artifactFolder).get();
  auto programInfo = compiled.getProgramInfo();
  auto keyset = getTestKeySetCachePtr()
                    ->getKeyset(programInfo.asReader().getKeyset(), 0, 0)
                    .value();
  auto csprng = std::make_shared<ConcreteCSPRNG>(0);

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
  auto compiled = engine.compile(sources, artifactFolder).get();
  auto programInfo = compiled.getProgramInfo();
  auto keyset = getTestKeySetCachePtr()
                    ->getKeyset(programInfo.asReader().getKeyset(), 0, 0)
                    .value();
  auto csprng = std::make_shared<ConcreteCSPRNG>(0);
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
      programInfo, compiled.getSharedLibraryPath(compiled.getOutputDirPath()),
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
    options.mainFuncName = "main";
    if (description.p_error) {
      assert(std::isnan(options.optimizerConfig.global_p_error));
      options.optimizerConfig.p_error = description.p_error.value();
    }
    options.optimizerConfig.encoding = description.encoding;
    auto context = mlir::concretelang::CompilationContext::createShared();
    mlir::concretelang::CompilerEngine engine(context);
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
              BM_Compile(st, description, engine, options);
            });
        break;
      case Action::KEYGEN:
        benchmark::RegisterBenchmark(
            benchName("keygen").c_str(), [=](::benchmark::State &st) {
              BM_KeyGen(st, description, engine, options);
            });
        break;
      case Action::ENCRYPT:
        benchmark::RegisterBenchmark(
            benchName("encrypt").c_str(), [=](::benchmark::State &st) {
              BM_ExportArguments(st, description, engine, options);
            });
        break;
      case Action::EVALUATE:
        benchmark::RegisterBenchmark(
            benchName("evaluate").c_str(), [=](::benchmark::State &st) {
              BM_Evaluate(st, description, engine, options);
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
  auto descriptionFiles = std::get<1>(options);

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
