#ifndef END_TO_END_TEST_H
#define END_TO_END_TEST_H

#include "concretelang/Support/CompilerEngine.h"
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
std::tuple<mlir::concretelang::CompilationOptions, std::string,
           std::vector<EndToEndDescFile>>
parseEndToEndCommandLine(int argc, char **argv) {

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
  llvm::cl::opt<std::optional<bool>> batchConcreteOps(
      "batch-concrete-ops",
      llvm::cl::desc(
          "Set the batchConcreteOps compilation options to run the tests"),
      llvm::cl::init(std::nullopt));

  // Optimizer options
  llvm::cl::opt<int> securityLevel(
      "security-level",
      llvm::cl::desc("Set the number of bit of security to target"),
      llvm::cl::init(mlir::concretelang::optimizer::DEFAULT_CONFIG.security));
  llvm::cl::opt<bool> optimizerDisplay(
      "optimizer-display",
      llvm::cl::desc("Set the optimizerConfig.display compilation options to "
                     "run the tests"),
      llvm::cl::init(false));

  // JIT or Library support
  llvm::cl::opt<bool> jit(
      "jit",
      llvm::cl::desc("Use JIT support to run the tests (default, overwritten "
                     "if --library is set"),
      llvm::cl::init(true));
  llvm::cl::opt<std::string> library(
      "library",
      llvm::cl::desc("Use library support to run the tests and specify the "
                     "prefix for compilation artifacts"),
      llvm::cl::init<std::string>(""));

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
  if (batchConcreteOps.has_value())
    compilationOptions.batchConcreteOps = batchConcreteOps.getValue().value();
  compilationOptions.optimizerConfig.display = optimizerDisplay.getValue();
  compilationOptions.optimizerConfig.security = securityLevel.getValue();

  std::vector<EndToEndDescFile> parsedDescriptionFiles;
  for (auto descFile : descriptionFiles) {
    EndToEndDescFile f;
    f.path = descFile;
    f.descriptions = loadEndToEndDesc(descFile);
    parsedDescriptionFiles.push_back(f);
  }
  auto libpath = library.getValue();
  if (libpath.empty() && !jit.getValue()) {
    llvm::errs()
        << "You must specify the library path or use jit to run the test";
    exit(1);
  }
  return std::make_tuple(compilationOptions, libpath, parsedDescriptionFiles);
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
  if (options.optimizerConfig.security != 128) {
    os << "_security" << options.optimizerConfig.security;
  }
  return os.str().substr(1);
}

#endif
