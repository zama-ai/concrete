#include "../end_to_end_tests/end_to_end_test.h"
#include "concretelang/Common/Compat.h"
#include "concretelang/TestLib/TestProgram.h"
#include <concretelang/Runtime/DFRuntime.hpp>

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
                       mlir::concretelang::CompilationOptions options) {
  TestProgram tc(options);
  for (auto _ : state) {
    assert(tc.compile(description.program));
  }
}

/// Benchmark time of the key generation
static void BM_KeyGen(benchmark::State &state, EndToEndDesc description,
                      mlir::concretelang::CompilationOptions options) {
  TestProgram tc(options);
  assert(tc.compile(description.program));

  for (auto _ : state) {
    assert(tc.generateKeyset(0, 0, false));
  }
}

/// Benchmark time of the encryption
static void BM_ExportArguments(benchmark::State &state,
                               EndToEndDesc description,
                               mlir::concretelang::CompilationOptions options) {
  TestProgram tc(options);
  assert(tc.compile(description.program));
  assert(tc.generateKeyset());

  assert(description.tests.size() > 0);
  auto test = description.tests[0];
  auto inputArguments = std::vector<TransportValue>();
  inputArguments.reserve(test.inputs.size());

  auto exporter = tc.getValueExporter().value();
  if (mlir::concretelang::dfr::_dfr_is_root_node()) {
    for (auto _ : state) {
      for (size_t i = 0; i < test.inputs.size(); i++) {
        auto input =
            exporter.prepareInput(test.inputs[i].getValue(), i).value();
        inputArguments.push_back(input);
      }
    }
    inputArguments.resize(0);
  }
}

/// Benchmark time of the program evaluation
static void BM_Evaluate(benchmark::State &state, EndToEndDesc description,
                        mlir::concretelang::CompilationOptions options) {
  TestProgram tc(options);
  assert(tc.compile(description.program));
  assert(tc.generateKeyset());
  auto exporter = tc.getValueExporter().value();

  assert(description.tests.size() > 0);
  auto test = description.tests[0];
  auto inputArguments = std::vector<TransportValue>();
  inputArguments.reserve(test.inputs.size());

  if (mlir::concretelang::dfr::_dfr_is_root_node()) {
    for (size_t i = 0; i < test.inputs.size(); i++) {
      auto input = exporter.prepareInput(test.inputs[i].getValue(), i).value();
      inputArguments.push_back(input);
    }
  }

  auto serverCircuit = tc.getServerCircuit().value();

  // Warmup
  assert(tc.callServer(inputArguments));

  for (auto _ : state) {
    assert(tc.callServer(inputArguments));
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
                               size_t stackSizeRequirement = 0,
                               int num_iterations = 0) {
  auto optionsName = getOptionsName(options);
  for (auto description : descriptions) {
    if (description.p_error) {
      assert(std::isnan(options.optimizerConfig.global_p_error));
      options.optimizerConfig.p_error = description.p_error.value();
    }
    options.optimizerConfig.encoding = description.encoding;
    auto benchName = [&](std::string name) {
      std::ostringstream s;
      s << suiteName << "/" << name << "/" << optionsName << "/"
        << description.description;
      return s.str();
    };
    for (auto action : actions) {
      switch (action) {
      case Action::COMPILE:
        benchmark::RegisterBenchmark(benchName("compile").c_str(),
                                     [=](::benchmark::State &st) {
                                       BM_Compile(st, description, options);
                                     });
        break;
      case Action::KEYGEN:
        benchmark::RegisterBenchmark(benchName("keygen").c_str(),
                                     [=](::benchmark::State &st) {
                                       BM_KeyGen(st, description, options);
                                     });
        break;
      case Action::ENCRYPT:
        benchmark::RegisterBenchmark(
            benchName("encrypt").c_str(), [=](::benchmark::State &st) {
              BM_ExportArguments(st, description, options);
            });
        break;
      case Action::EVALUATE:
        auto bench = benchmark::RegisterBenchmark(
            benchName("evaluate").c_str(), [=](::benchmark::State &st) {
              BM_Evaluate(st, description, options);
            });
        if (num_iterations)
          bench->Iterations(num_iterations);
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
                              std::get<0>(options).compilationOptions, actions,
                              stackSizeRequirement,
                              std::get<0>(options).numIterations);
  }
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  _dfr_terminate();
  return 0;
}
