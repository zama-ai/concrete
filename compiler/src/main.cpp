// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cstdint>
#include <iostream>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/ToolUtilities.h>
#include <sstream>

#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/ClientLib/KeySetCache.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Dialect/RT/IR/RTDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"
#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/JITSupport.h"
#include "concretelang/Support/LLVMEmitFile.h"
#include "concretelang/Support/Pipeline.h"
#include "concretelang/Support/V0Parameters.h"
#include "concretelang/Support/logging.h"
#include "mlir/IR/BuiltinOps.h"

namespace clientlib = concretelang::clientlib;

enum Action {
  ROUND_TRIP,
  DUMP_FHE,
  DUMP_TFHE,
  DUMP_CONCRETE,
  DUMP_CONCRETEWITHLOOPS,
  DUMP_BCONCRETE,
  DUMP_SDFG,
  DUMP_STD,
  DUMP_LLVM_DIALECT,
  DUMP_LLVM_IR,
  DUMP_OPTIMIZED_LLVM_IR,
  JIT_INVOKE,
  COMPILE,
};

namespace cmdline {
const std::string STDOUT = "-";
class OptionalSizeTParser : public llvm::cl::parser<llvm::Optional<size_t>> {
public:
  OptionalSizeTParser(llvm::cl::Option &option)
      : llvm::cl::parser<llvm::Optional<size_t>>(option) {}

  bool parse(llvm::cl::Option &option, llvm::StringRef argName,
             llvm::StringRef arg, llvm::Optional<size_t> &value) {
    size_t parsedVal;
    std::istringstream iss(arg.str());

    iss >> parsedVal;

    if (iss.fail())
      return option.error("Invalid value " + arg);

    value.emplace(parsedVal);

    return false;
  }
};

llvm::cl::list<std::string> inputs(llvm::cl::Positional,
                                   llvm::cl::desc("<Input files>"),
                                   llvm::cl::OneOrMore);

llvm::cl::opt<std::string> output("o",
                                  llvm::cl::desc("Specify output filename"),
                                  llvm::cl::value_desc("filename"),
                                  llvm::cl::init(STDOUT));

llvm::cl::opt<bool> verbose("verbose", llvm::cl::desc("verbose logs"),
                            llvm::cl::init<bool>(false));

llvm::cl::opt<bool>
    optimizeConcrete("optimize-concrete",
                     llvm::cl::desc("enable/disable optimizations of concrete "
                                    "dialects. (Enabled by default)"),
                     llvm::cl::init<bool>(true));

llvm::cl::opt<bool> emitGPUOps(
    "emit-gpu-ops",
    llvm::cl::desc(
        "enable/disable generating GPU operations (Disabled by default)"),
    llvm::cl::init<bool>(false));

llvm::cl::list<std::string> passes(
    "passes",
    llvm::cl::desc("Specify the passes to run (use only for compiler tests)"),
    llvm::cl::value_desc("passname"), llvm::cl::ZeroOrMore);

static llvm::cl::opt<enum Action> action(
    "a", "action", llvm::cl::desc("output mode"), llvm::cl::ValueRequired,
    llvm::cl::NumOccurrencesFlag::Required,
    llvm::cl::values(
        clEnumValN(Action::ROUND_TRIP, "roundtrip",
                   "Parse input module and regenerate textual representation")),
    llvm::cl::values(clEnumValN(Action::DUMP_FHE, "dump-fhe",
                                "Dump FHE module")),
    llvm::cl::values(clEnumValN(Action::DUMP_TFHE, "dump-tfhe",
                                "Lower to TFHE and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_CONCRETE, "dump-concrete",
                                "Lower to Concrete and dump result")),
    llvm::cl::values(clEnumValN(
        Action::DUMP_CONCRETEWITHLOOPS, "dump-concrete-with-loops",
        "Lower to Concrete, replace linalg ops with loops and dump result")),
    llvm::cl::values(
        clEnumValN(Action::DUMP_BCONCRETE, "dump-bconcrete",
                   "Lower to Bufferized Concrete and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_SDFG, "dump-sdfg",
                                "Lower to SDFG operations annd dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_STD, "dump-std",
                                "Lower to std and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_LLVM_DIALECT, "dump-llvm-dialect",
                                "Lower to LLVM dialect and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_LLVM_IR, "dump-llvm-ir",
                                "Lower to LLVM-IR and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_OPTIMIZED_LLVM_IR,
                                "dump-optimized-llvm-ir",
                                "Lower to LLVM-IR, optimize and dump result")),
    llvm::cl::values(clEnumValN(Action::JIT_INVOKE, "jit-invoke",
                                "Lower and JIT-compile input module and invoke "
                                "function specified with --funcname")),
    llvm::cl::values(clEnumValN(Action::COMPILE, "compile",
                                "Lower to LLVM-IR, compile to a file")));

llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

llvm::cl::opt<bool> autoParallelize(
    "parallelize",
    llvm::cl::desc("Generate (and execute if JIT) parallel code"),
    llvm::cl::init(false));

llvm::cl::opt<bool> loopParallelize(
    "parallelize-loops",
    llvm::cl::desc(
        "Generate (and execute if JIT) parallel loops from Linalg operations"),
    llvm::cl::init(false));

llvm::cl::opt<bool> batchConcreteOps(
    "batch-concrete-ops",
    llvm::cl::desc(
        "Hoist scalar Concrete operations with corresponding batched "
        "operations out of loop nests as batched operations"),
    llvm::cl::init(false));

llvm::cl::opt<bool> emitSDFGOps(
    "emit-sdfg-ops",
    llvm::cl::desc(
        "Extract operations supported by the SDFG dialect for static data flow"
        " graphs and emit them."),
    llvm::cl::init(false));

llvm::cl::opt<bool> dataflowParallelize(
    "parallelize-dataflow",
    llvm::cl::desc(
        "Generate (and execute if JIT) the program as a dataflow graph"),
    llvm::cl::init(false));

llvm::cl::opt<std::string>
    funcName("funcname",
             llvm::cl::desc("Name of the function to compile, default 'main'"),
             llvm::cl::init<std::string>(""));

llvm::cl::list<uint64_t>
    jitArgs("jit-args",
            llvm::cl::desc("Value of arguments to pass to the main func"),
            llvm::cl::value_desc("argument(uint64)"), llvm::cl::ZeroOrMore,
            llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::opt<std::string> jitKeySetCachePath(
    "jit-keyset-cache-path",
    llvm::cl::desc("Path to cache KeySet content (unsecure)"));

llvm::cl::opt<double> pbsErrorProbability(
    "pbs-error-probability",
    llvm::cl::desc("Change the default probability of error for all pbs"),
    llvm::cl::init(mlir::concretelang::optimizer::DEFAULT_CONFIG.p_error));

llvm::cl::opt<double> globalErrorProbability(
    "global-error-probability",
    llvm::cl::desc(
        "Use global error probability (override pbs error probability)"),
    llvm::cl::init(
        mlir::concretelang::optimizer::DEFAULT_CONFIG.global_p_error));

llvm::cl::opt<bool> displayOptimizerChoice(
    "display-optimizer-choice",
    llvm::cl::desc("Display the information returned by the optimizer"),
    llvm::cl::init(false));

llvm::cl::opt<bool>
    optimizerV0("optimizer-v0",
                llvm::cl::desc("Select the v0 parameters strategy"),
                llvm::cl::init(false));

llvm::cl::opt<double> fallbackLogNormWoppbs(
    "optimizer-fallback-log-norm-woppbs",
    llvm::cl::desc("Select a fallback value for multisum log norm in woppbs "
                   "when the precise value can't be computed."),
    llvm::cl::init(mlir::concretelang::optimizer::DEFAULT_CONFIG
                       .fallback_log_norm_woppbs));

llvm::cl::opt<concrete_optimizer::Encoding> optimizerEncoding(
    "force-encoding", llvm::cl::desc("Choose cyphertext encoding."),
    llvm::cl::init(mlir::concretelang::optimizer::DEFAULT_CONFIG.encoding),
    llvm::cl::values(clEnumValN(concrete_optimizer::Encoding::Auto, "auto",
                                "Pick the best [default]")),
    llvm::cl::values(clEnumValN(concrete_optimizer::Encoding::Native, "native",
                                "native")),
    llvm::cl::values(clEnumValN(concrete_optimizer::Encoding::Native, "crt",
                                "Chineese Reminder Theorem representation")));

llvm::cl::list<int64_t> fhelinalgTileSizes(
    "fhelinalg-tile-sizes",
    llvm::cl::desc(
        "Force tiling of FHELinalg operation with the given tile sizes"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::list<size_t> v0Constraint(
    "v0-constraint",
    llvm::cl::desc(
        "Force the compiler to use the given v0 constraint [p, norm2]"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::list<size_t> v0Parameter(
    "v0-parameter",
    llvm::cl::desc(
        "Force to apply the given v0 parameters [glweDimension, "
        "logPolynomialSize, nSmall, brLevel, brLobBase, ksLevel, ksLogBase]"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::list<int64_t> largeIntegerCRTDecomposition(
    "large-integer-crt-decomposition",
    llvm::cl::desc(
        "Use the large integer to lower FHE.eint with the given decomposition, "
        "must be used with the other large-integers options (experimental)"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::list<int64_t> largeIntegerPackingKeyswitch(
    "large-integer-packing-keyswitch",
    llvm::cl::desc(
        "Use the large integer to lower FHE.eint with the given parameters for "
        "packing keyswitch, must be used with the other large-integers options "
        "(experimental) [inputLweDimension, outputPolynomialSize, level, "
        "baseLog]"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::list<int64_t> largeIntegerCircuitBootstrap(
    "large-integer-circuit-bootstrap",
    llvm::cl::desc(
        "Use the large integer to lower FHE.eint with the given parameters for "
        "the cicuit boostrap, must be used with the other large-integers "
        "options "
        "(experimental) [level, baseLog]"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

} // namespace cmdline

namespace llvm {
// This needs to be wrapped into the llvm namespace for proper
// operator lookup
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const llvm::ArrayRef<uint64_t> arr) {
  os << "(";
  for (size_t i = 0; i < arr.size(); i++) {
    os << arr[i];

    if (i != arr.size() - 1)
      os << ", ";
  }

  return os;
}
} // namespace llvm

llvm::Expected<mlir::concretelang::CompilationOptions>
cmdlineCompilationOptions() {
  mlir::concretelang::CompilationOptions options;

  options.verifyDiagnostics = cmdline::verifyDiagnostics;
  options.autoParallelize = cmdline::autoParallelize;
  options.loopParallelize = cmdline::loopParallelize;
  options.dataflowParallelize = cmdline::dataflowParallelize;
  options.batchConcreteOps = cmdline::batchConcreteOps;
  options.emitSDFGOps = cmdline::emitSDFGOps;
  options.optimizeConcrete = cmdline::optimizeConcrete;
  options.emitGPUOps = cmdline::emitGPUOps;

  if (!cmdline::v0Constraint.empty()) {
    if (cmdline::v0Constraint.size() != 2) {
      return llvm::make_error<llvm::StringError>(
          "The v0-constraint option expect a list of size 2",
          llvm::inconvertibleErrorCode());
    }
    options.v0FHEConstraints = mlir::concretelang::V0FHEConstraint{
        cmdline::v0Constraint[1], cmdline::v0Constraint[0]};
  }

  if (!cmdline::funcName.empty()) {
    options.clientParametersFuncName = cmdline::funcName;
  }

  // Convert tile sizes to `Optional`
  if (!cmdline::fhelinalgTileSizes.empty())
    options.fhelinalgTileSizes.emplace(cmdline::fhelinalgTileSizes);

  // Setup the v0 parameter options
  if (!cmdline::v0Parameter.empty()) {
    if (cmdline::v0Parameter.size() != 7) {
      return llvm::make_error<llvm::StringError>(
          "The v0-parameter option expect a list of size 7",
          llvm::inconvertibleErrorCode());
    }
    options.v0Parameter = {cmdline::v0Parameter[0], cmdline::v0Parameter[1],
                           cmdline::v0Parameter[2], cmdline::v0Parameter[3],
                           cmdline::v0Parameter[4], cmdline::v0Parameter[5],
                           cmdline::v0Parameter[6], llvm::None};
  }

  // Setup the large integer options
  if (!cmdline::largeIntegerCRTDecomposition.empty() ||
      !cmdline::largeIntegerPackingKeyswitch.empty() ||
      !cmdline::largeIntegerPackingKeyswitch.empty()) {
    if (cmdline::largeIntegerCRTDecomposition.empty() ||
        cmdline::largeIntegerPackingKeyswitch.empty() ||
        cmdline::largeIntegerPackingKeyswitch.empty()) {
      return llvm::make_error<llvm::StringError>(
          "The large-integers options should all be set",
          llvm::inconvertibleErrorCode());
    }
    if (cmdline::largeIntegerPackingKeyswitch.size() != 5) {
      return llvm::make_error<llvm::StringError>(
          "The large-integers-packing-keyswitch must be a list of 5 integer",
          llvm::inconvertibleErrorCode());
    }
    if (cmdline::largeIntegerCircuitBootstrap.size() != 2) {
      return llvm::make_error<llvm::StringError>(
          "The large-integers-packing-keyswitch must be a list of 2 integer",
          llvm::inconvertibleErrorCode());
    }
    options.largeIntegerParameter = mlir::concretelang::LargeIntegerParameter();
    options.largeIntegerParameter->crtDecomposition =
        cmdline::largeIntegerCRTDecomposition;
    options.largeIntegerParameter->wopPBS.packingKeySwitch.inputLweDimension =
        cmdline::largeIntegerPackingKeyswitch[0];
    options.largeIntegerParameter->wopPBS.packingKeySwitch
        .outputPolynomialSize = cmdline::largeIntegerPackingKeyswitch[1];
    options.largeIntegerParameter->wopPBS.packingKeySwitch.level =
        cmdline::largeIntegerPackingKeyswitch[2];
    options.largeIntegerParameter->wopPBS.packingKeySwitch.baseLog =
        cmdline::largeIntegerPackingKeyswitch[3];
    options.largeIntegerParameter->wopPBS.circuitBootstrap.level =
        cmdline::largeIntegerCircuitBootstrap[0];
    options.largeIntegerParameter->wopPBS.circuitBootstrap.baseLog =
        cmdline::largeIntegerCircuitBootstrap[1];
  }

  options.optimizerConfig.global_p_error = cmdline::globalErrorProbability;
  options.optimizerConfig.p_error = cmdline::pbsErrorProbability;
  options.optimizerConfig.display = cmdline::displayOptimizerChoice;
  options.optimizerConfig.strategy_v0 = cmdline::optimizerV0;
  options.optimizerConfig.encoding = cmdline::optimizerEncoding;

  if (!std::isnan(options.optimizerConfig.global_p_error) &&
      options.optimizerConfig.strategy_v0) {
    return llvm::make_error<llvm::StringError>(
        "--global-error-probability is not compatible with --optimizer-v0",
        llvm::inconvertibleErrorCode());
  }

  return options;
}

/// Process a single source buffer
///
/// The parameter `action` specifies how the buffer should be processed
/// and thus defines the output.
///
/// If the specified action involves JIT compilation, `funcName`
/// designates the function to JIT compile. This function is invoked
/// using the parameters given in `jitArgs`.
///
/// The parameter `parametrizeTFHE` defines, whether the
/// parametrization pass for TFHE is executed. If the `action` does
/// not involve any MidlFHE manipulation, this parameter does not have
/// any effect.
///
/// The parameters `overrideMaxEintPrecision` and `overrideMaxMANP`, if
/// set, override the values for the maximum required precision of
/// encrypted integers and the maximum value for the Minimum Arithmetic
/// Noise Padding otherwise determined automatically.
///
/// If `verifyDiagnostics` is `true`, the procedure only checks if the
/// diagnostic messages provided in the source buffer using
/// `expected-error` are produced. If `verifyDiagnostics` is `false`,
/// the procedure checks if the parsed module is valid and if all
/// requested transformations succeeded.
///
/// Compilation output is written to the stream specified by `os`.
mlir::LogicalResult processInputBuffer(
    std::unique_ptr<llvm::MemoryBuffer> buffer, std::string sourceFileName,
    mlir::concretelang::CompilationOptions &options, enum Action action,
    llvm::ArrayRef<uint64_t> jitArgs,
    llvm::Optional<clientlib::KeySetCache> keySetCache, llvm::raw_ostream &os,
    std::shared_ptr<mlir::concretelang::CompilerEngine::Library> outputLib) {
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();

  std::string funcName = options.clientParametersFuncName.getValueOr("");
  if (action == Action::JIT_INVOKE) {
    auto lambdaOrErr =
        mlir::concretelang::ClientServer<mlir::concretelang::JITSupport>::
            create(buffer->getBuffer(), options, keySetCache,
                   mlir::concretelang::JITSupport());
    if (!lambdaOrErr) {
      mlir::concretelang::log_error()
          << "Failed to get JIT-lambda " << funcName << " "
          << llvm::toString(lambdaOrErr.takeError());
      return mlir::failure();
    }
    llvm::Expected<uint64_t> resOrErr = (*lambdaOrErr)(jitArgs);
    if (!resOrErr) {
      mlir::concretelang::log_error()
          << "Failed to JIT-invoke " << funcName << " with arguments "
          << jitArgs << ": " << llvm::toString(resOrErr.takeError());
      return mlir::failure();
    }

    os << *resOrErr << "\n";
  } else {
    mlir::concretelang::CompilerEngine ce{ccx};
    ce.setCompilationOptions(options);

    if (cmdline::passes.size() != 0) {
      ce.setEnablePass([](mlir::Pass *pass) {
        return std::any_of(
            cmdline::passes.begin(), cmdline::passes.end(),
            [&](const std::string &p) { return pass->getArgument() == p; });
      });
    }
    enum mlir::concretelang::CompilerEngine::Target target;

    switch (action) {
    case Action::ROUND_TRIP:
      target = mlir::concretelang::CompilerEngine::Target::ROUND_TRIP;
      break;
    case Action::DUMP_FHE:
      target = mlir::concretelang::CompilerEngine::Target::FHE;
      break;
    case Action::DUMP_TFHE:
      target = mlir::concretelang::CompilerEngine::Target::TFHE;
      break;
    case Action::DUMP_CONCRETE:
      target = mlir::concretelang::CompilerEngine::Target::CONCRETE;
      break;
    case Action::DUMP_CONCRETEWITHLOOPS:
      target = mlir::concretelang::CompilerEngine::Target::CONCRETEWITHLOOPS;
      break;
    case Action::DUMP_BCONCRETE:
      target = mlir::concretelang::CompilerEngine::Target::BCONCRETE;
      break;
    case Action::DUMP_SDFG:
      target = mlir::concretelang::CompilerEngine::Target::SDFG;
      break;
    case Action::DUMP_STD:
      target = mlir::concretelang::CompilerEngine::Target::STD;
      break;
    case Action::DUMP_LLVM_DIALECT:
      target = mlir::concretelang::CompilerEngine::Target::LLVM;
      break;
    case Action::DUMP_LLVM_IR:
      target = mlir::concretelang::CompilerEngine::Target::LLVM_IR;
      break;
    case Action::DUMP_OPTIMIZED_LLVM_IR:
      target = mlir::concretelang::CompilerEngine::Target::OPTIMIZED_LLVM_IR;
      break;
    case Action::COMPILE:
      target = mlir::concretelang::CompilerEngine::Target::LIBRARY;
      break;
    case JIT_INVOKE:
      // Case just here to satisfy the compiler; already handled above
      abort();
      break;
    }
    auto retOrErr = ce.compile(std::move(buffer), target, outputLib);

    if (!retOrErr) {
      mlir::concretelang::log_error()
          << llvm::toString(retOrErr.takeError()) << "\n";

      return mlir::failure();
    }

    if (retOrErr->llvmModule) {
      // At least usefull for intermediate binary object files naming
      retOrErr->llvmModule->setSourceFileName(sourceFileName);
      retOrErr->llvmModule->setModuleIdentifier(sourceFileName);
    }

    if (options.verifyDiagnostics) {
      return mlir::success();
    } else if (action == Action::DUMP_LLVM_IR ||
               action == Action::DUMP_OPTIMIZED_LLVM_IR) {
      retOrErr->llvmModule->print(os, nullptr);
    } else if (action != Action::COMPILE) {
      retOrErr->mlirModuleRef->get().print(os);
    }
  }

  return mlir::success();
}

mlir::LogicalResult compilerMain(int argc, char **argv) {
  // Parse command line arguments
  llvm::cl::ParseCommandLineOptions(argc, argv);

  mlir::concretelang::setupLogging(cmdline::verbose);

  // String for error messages
  std::string errorMessage;

  if (cmdline::action == Action::COMPILE) {
    if (cmdline::output == cmdline::STDOUT) {
      // can't use stdin to generate a lib.
      errorMessage += "Please provide a file destination '-o' option.\n";
    }
    // SplitInputFile would need to have separate object files
    // destinations to be able to work.
    if (cmdline::splitInputFile) {
      errorMessage +=
          "'--action=compile' and '--split-input-file' are incompatible\n";
    }
    if (errorMessage != "") {
      llvm::errs() << errorMessage << "\n";
      return mlir::failure();
    }
  }

  auto compilerOptions = cmdlineCompilationOptions();
  if (auto err = compilerOptions.takeError()) {
    llvm::errs() << err << "\n";
    return mlir::failure();
  }

  llvm::Optional<clientlib::KeySetCache> jitKeySetCache;
  if (!cmdline::jitKeySetCachePath.empty()) {
    jitKeySetCache = clientlib::KeySetCache(cmdline::jitKeySetCachePath);
  }

  // In case of compilation to library, the real output is the library.
  std::string outputPath =
      (cmdline::action == Action::COMPILE) ? cmdline::STDOUT : cmdline::output;

  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile(outputPath, &errorMessage);

  using Library = mlir::concretelang::CompilerEngine::Library;
  auto outputLib = std::make_shared<Library>(cmdline::output);

  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }

  // Iterate over all input files specified on the command line
  for (const auto &fileName : cmdline::inputs) {
    auto file = mlir::openInputFile(fileName, &errorMessage);

    if (!file) {
      llvm::errs() << errorMessage << "\n";
      return mlir::failure();
    }

    // If `--split-input-file` is set, the file is split into
    // individual chunks separated by `// -----` markers. Each chunk
    // is then processed individually as if it were part of a separate
    // source file.
    auto process = [&](std::unique_ptr<llvm::MemoryBuffer> inputBuffer,
                       llvm::raw_ostream &os) {
      return processInputBuffer(
          std::move(inputBuffer), fileName, *compilerOptions, cmdline::action,
          cmdline::jitArgs, jitKeySetCache, os, outputLib);
    };
    auto &os = output->os();
    auto res = mlir::failure();
    if (cmdline::splitInputFile) {
      res = mlir::splitAndProcessBuffer(std::move(file), process, os);
    } else {
      res = process(std::move(file), os);
    }
    if (res.failed()) {
      return mlir::failure();
    } else {
      output->keep();
    }
  }

  if (cmdline::action == Action::COMPILE) {
    auto err = outputLib->emitArtifacts(
        /*sharedLib=*/true, /*staticLib=*/true,
        /*clientParameters=*/true, /*compilationFeedback=*/true,
        /*cppHeader=*/true);
    if (err) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  int result = 0;
  if (mlir::failed(compilerMain(argc, argv)))
    result = 1;

  _dfr_terminate();
  return result;
}
