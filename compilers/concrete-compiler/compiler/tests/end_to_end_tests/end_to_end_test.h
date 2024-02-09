#ifndef END_TO_END_TEST_H
#define END_TO_END_TEST_H

#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/logging.h"
#include "llvm/Support/CommandLine.h"

#include "end_to_end_fixture/EndToEndFixture.h"

// Shorthands to create integer literals of a specific type
static inline uint8_t operator"" _u8(unsigned long long int v) { return v; }
static inline uint16_t operator"" _u16(unsigned long long int v) { return v; }
static inline uint32_t operator"" _u32(unsigned long long int v) { return v; }
static inline uint64_t operator"" _u64(unsigned long long int v) { return v; }

const double TEST_ERROR_RATE = 1.0 - 0.999936657516;

// Evaluates to the number of elements of a statically initialized
// array
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

typedef struct EndToEndTestOptions {
  mlir::concretelang::CompilationOptions compilationOptions;
  int numberOfRetry;
  int numIterations;
} EndToEndTestOptions;

/// @brief  Parse the command line and return a tuple contains the compilation
/// options, the library path if the --library options has been specified and
/// the parsed description files
std::pair<EndToEndTestOptions, std::vector<EndToEndDescFile>>
parseEndToEndCommandLine(int argc, char **argv) {
  namespace optimizer = mlir::concretelang::optimizer;
  // TODO - Well reset other llvm command line options registered but assert on
  // --help
  // llvm::cl::ResetCommandLineParser();

  llvm::cl::list<std::string> descriptionFiles(
      llvm::cl::Positional, llvm::cl::desc("<End to end description Files>"),
      llvm::cl::OneOrMore);

  // Compilation options
  llvm::cl::opt<mlir::concretelang::Backend> backend(
      "backend",
      llvm::cl::desc("Specify benchmark cases to run, if no benchmarks speci"),
      llvm::cl::values(clEnumValN(mlir::concretelang::Backend::CPU, "cpu",
                                  "Target a CPU backend")),
      llvm::cl::values(clEnumValN(mlir::concretelang::Backend::GPU, "gpu",
                                  "Target a GPU backend")),
      llvm::cl::init(mlir::concretelang::Backend::CPU));

  llvm::cl::opt<int> loopParallelize(
      "loop-parallelize",
      llvm::cl::desc(
          "Set the loopParallelize compilation options to run the tests"),
      llvm::cl::init(-1));
  llvm::cl::opt<int> dataflowParallelize(
      "dataflow-parallelize",
      llvm::cl::desc(
          "Set the loopParallelize compilation options to run the tests"),
      llvm::cl::init(-1));
  llvm::cl::opt<int> emitGPUOps(
      "emit-gpu-ops",
      llvm::cl::desc("Set the emitGPUOps compilation options to run the tests"),
      llvm::cl::init(-1));
  llvm::cl::opt<int> batchTFHEOps(
      "batch-tfhe-ops",
      llvm::cl::desc(
          "Set the batchTFHEOps compilation options to run the tests"),
      llvm::cl::init(-1));
  llvm::cl::opt<bool> simulate("simulate",
                               llvm::cl::desc("Simulate the FHE execution"),
                               llvm::cl::init(false));
  llvm::cl::opt<bool> compressEvaluationKeys(
      "compress-evaluation-keys",
      llvm::cl::desc("Enable the compression of evaluation keys"),
      llvm::cl::init(false));

  llvm::cl::opt<bool> distBenchmark(
      "distributed",
      llvm::cl::desc("Force a constant number of iterations in the benchmark "
                     "suite as required for distributed execution (default: 1 "
                     "- use --iterations=<n> to change)"),
      llvm::cl::init(false));
  llvm::cl::opt<int> numIterations(
      "iterations",
      llvm::cl::desc("Set the number of iterations for the benchmark suite "
                     "(only to be used with --distributed)"),
      llvm::cl::init(1));

  // Optimizer options
  llvm::cl::opt<int> securityLevel(
      "security-level",
      llvm::cl::desc("Set the number of bit of security to target"),
      llvm::cl::init(optimizer::DEFAULT_CONFIG.security));
  llvm::cl::opt<bool> optimizerDisplay(
      "optimizer-display",
      llvm::cl::desc("Set the optimizerConfig.display compilation options to "
                     "run the tests"),
      llvm::cl::init(false));
  llvm::cl::opt<optimizer::Strategy> optimizerStrategy(
      "optimizer-strategy",
      llvm::cl::desc("Select the concrete optimizer strategy"),
      llvm::cl::init(optimizer::DEFAULT_STRATEGY),
      llvm::cl::values(clEnumValN(optimizer::Strategy::V0, "V0",
                                  "Use the V0 optimizer strategy that use the "
                                  "worst case atomic pattern")),
      llvm::cl::values(clEnumValN(
          optimizer::Strategy::DAG_MONO, "dag-mono",
          "problem using the fhe computation dag with ONE set of evaluation "
          "keys")),
      llvm::cl::values(clEnumValN(
          optimizer::Strategy::DAG_MULTI, "dag-multi",
          "Use the dag-multi optimizer strategy that solve the optimization "
          "problem using the fhe computation dag with SEVERAL set of "
          "evaluation "
          "keys")));
  llvm::cl::opt<bool> keySharing(
      "optimizer-key-sharing",
      llvm::cl::desc(
          "Set the optimizerConfig.key_sharing compilation options to "
          "run the tests"),
      llvm::cl::init(true));

  // Verbose compiler
  llvm::cl::opt<bool> verbose("verbose",
                              llvm::cl::desc("Set the compiler verbosity"),
                              llvm::cl::init(false));

  // e2e test options
  llvm::cl::opt<int> retryFailingTests("retry-failing-tests",
                                       llvm::cl::desc("Retry test which fails"),
                                       llvm::cl::init(0));

  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Build compilation options
  mlir::concretelang::CompilationOptions compilationOptions(backend.getValue());
  if (loopParallelize.getValue() != -1)
    compilationOptions.loopParallelize = loopParallelize.getValue();

  if (dataflowParallelize.getValue() != -1)
    compilationOptions.dataflowParallelize = dataflowParallelize.getValue();
  if (emitGPUOps.getValue() != -1)
    compilationOptions.emitGPUOps = emitGPUOps.getValue();
  if (batchTFHEOps.getValue() != -1)
    compilationOptions.batchTFHEOps = batchTFHEOps.getValue();
  compilationOptions.simulate = simulate.getValue();
  compilationOptions.compressEvaluationKeys = compressEvaluationKeys.getValue();
  compilationOptions.optimizerConfig.display = optimizerDisplay.getValue();
  compilationOptions.optimizerConfig.security = securityLevel.getValue();
  compilationOptions.optimizerConfig.strategy = optimizerStrategy.getValue();
  compilationOptions.optimizerConfig.key_sharing = keySharing.getValue();
  mlir::concretelang::setupLogging(verbose.getValue());

  std::vector<EndToEndDescFile> parsedDescriptionFiles;
  for (auto descFile : descriptionFiles) {
    EndToEndDescFile f;
    f.path = descFile;
    f.descriptions = loadEndToEndDesc(descFile);
    parsedDescriptionFiles.push_back(f);
  }
  int num_iterations =
      (distBenchmark.getValue()) ? numIterations.getValue() : 0;
  return std::make_pair(
      EndToEndTestOptions{
          compilationOptions,
          retryFailingTests.getValue(),
          num_iterations,
      },
      parsedDescriptionFiles);
}

std::string getOptionsName(mlir::concretelang::CompilationOptions compilation) {
  namespace optimizer = mlir::concretelang::optimizer;
  std::ostringstream os;
  if (compilation.simulate)
    os << "_simulate";
  if (compilation.loopParallelize)
    os << "_loop";
  if (compilation.dataflowParallelize)
    os << "_dataflow";
  if (compilation.emitGPUOps)
    os << "_gpu";
  auto ostr = os.str();
  if (ostr.size() == 0) {
    os << "_default";
  }
  if (compilation.optimizerConfig.security !=
      optimizer::DEFAULT_CONFIG.security) {
    os << "_security" << compilation.optimizerConfig.security;
  }
  if (compilation.optimizerConfig.strategy !=
      optimizer::DEFAULT_CONFIG.strategy) {
    os << "_" << compilation.optimizerConfig.strategy;
  }
  return os.str().substr(1);
}

#endif
