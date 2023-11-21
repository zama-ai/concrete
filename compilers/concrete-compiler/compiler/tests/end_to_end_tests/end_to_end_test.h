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

/// @brief  Parse the command line and return a tuple contains the compilation
/// options, the library path if the --library options has been specified and
/// the parsed description files
std::tuple<mlir::concretelang::CompilationOptions,
           std::vector<EndToEndDescFile>, int>
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

  llvm::cl::opt<std::optional<bool>> loopParallelize(
      "loop-parallelize",
      llvm::cl::desc(
          "Set the loopParallelize compilation options to run the tests"),
      llvm::cl::init(std::nullopt));
  llvm::cl::opt<std::optional<bool>> dataflowParallelize(
      "dataflow-parallelize",
      llvm::cl::desc(
          "Set the loopParallelize compilation options to run the tests"),
      llvm::cl::init(std::nullopt));
  llvm::cl::opt<std::optional<bool>> emitGPUOps(
      "emit-gpu-ops",
      llvm::cl::desc("Set the emitGPUOps compilation options to run the tests"),
      llvm::cl::init(std::nullopt));
  llvm::cl::opt<std::optional<bool>> batchTFHEOps(
      "batch-tfhe-ops",
      llvm::cl::desc(
          "Set the batchTFHEOps compilation options to run the tests"),
      llvm::cl::init(std::nullopt));

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
      llvm::cl::values(clEnumValN(optimizer::Strategy::V0,
                                  toString(optimizer::Strategy::V0),
                                  "Use the V0 optimizer strategy that use the "
                                  "worst case atomic pattern")),
      llvm::cl::values(clEnumValN(
          optimizer::Strategy::DAG_MONO,
          toString(optimizer::Strategy::DAG_MONO),
          "Use the dag-mono optimizer strategy that solve the optimization "
          "problem using the fhe computation dag with ONE set of evaluation "
          "keys")),
      llvm::cl::values(clEnumValN(
          optimizer::Strategy::DAG_MULTI,
          toString(optimizer::Strategy::DAG_MULTI),
          "Use the dag-multi optimizer strategy that solve the optimization "
          "problem using the fhe computation dag with SEVERAL set of "
          "evaluation "
          "keys")));

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
  mlir::concretelang::CompilationOptions compilationOptions("main",
                                                            backend.getValue());
  if (loopParallelize.has_value())
    compilationOptions.loopParallelize = loopParallelize.getValue().value();
  if (dataflowParallelize.has_value())
    compilationOptions.dataflowParallelize =
        dataflowParallelize.getValue().value();
  if (emitGPUOps.has_value())
    compilationOptions.emitGPUOps = emitGPUOps.getValue().value();
  if (batchTFHEOps.has_value())
    compilationOptions.batchTFHEOps = batchTFHEOps.getValue().value();
  compilationOptions.optimizerConfig.display = optimizerDisplay.getValue();
  compilationOptions.optimizerConfig.security = securityLevel.getValue();
  compilationOptions.optimizerConfig.strategy = optimizerStrategy.getValue();

  mlir::concretelang::setupLogging(verbose.getValue());

  std::vector<EndToEndDescFile> parsedDescriptionFiles;
  for (auto descFile : descriptionFiles) {
    EndToEndDescFile f;
    f.path = descFile;
    f.descriptions = loadEndToEndDesc(descFile);
    parsedDescriptionFiles.push_back(f);
  }

  return std::make_tuple(compilationOptions, parsedDescriptionFiles,
                         retryFailingTests.getValue());
}

std::string getOptionsName(mlir::concretelang::CompilationOptions options) {
  namespace optimizer = mlir::concretelang::optimizer;
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
  if (options.optimizerConfig.security != optimizer::DEFAULT_CONFIG.security) {
    os << "_security" << options.optimizerConfig.security;
  }
  if (options.optimizerConfig.strategy != optimizer::DEFAULT_CONFIG.strategy) {
    os << "_" << options.optimizerConfig.strategy;
  }
  return os.str().substr(1);
}

#endif
