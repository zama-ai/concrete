// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <cstdint>
#include <iostream>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/ToolUtilities.h>
#include <sstream>

#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Dialect/RT/IR/RTDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/JitCompilerEngine.h"
#include "concretelang/Support/LLVMEmitFile.h"
#include "concretelang/Support/Pipeline.h"
#include "concretelang/Support/logging.h"
#include "mlir/IR/BuiltinOps.h"

enum Action {
  ROUND_TRIP,
  DUMP_FHE,
  DUMP_TFHE,
  DUMP_CONCRETE,
  DUMP_BCONCRETE,
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
    llvm::cl::values(
        clEnumValN(Action::DUMP_BCONCRETE, "dump-bconcrete",
                   "Lower to Bufferized Concrete and dump result")),
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

llvm::cl::opt<bool> dataflowParallelize(
    "parallelize-dataflow",
    llvm::cl::desc(
        "Generate (and execute if JIT) the program as a dataflow graph"),
    llvm::cl::init(false));

llvm::cl::opt<std::string>
    funcName("funcname",
             llvm::cl::desc("Name of the function to compile, default 'main'"),
             llvm::cl::init<std::string>("main"));

llvm::cl::list<uint64_t>
    jitArgs("jit-args",
            llvm::cl::desc("Value of arguments to pass to the main func"),
            llvm::cl::value_desc("argument(uint64)"), llvm::cl::ZeroOrMore);

llvm::cl::opt<std::string> jitKeySetCachePath(
    "jit-keyset-cache-path",
    llvm::cl::desc("Path to cache KeySet content (unsecure)"));

llvm::cl::opt<llvm::Optional<size_t>, false, OptionalSizeTParser>
    assumeMaxEintPrecision(
        "assume-max-eint-precision",
        llvm::cl::desc("Assume a maximum precision for encrypted integers"));

llvm::cl::opt<llvm::Optional<size_t>, false, OptionalSizeTParser> assumeMaxMANP(
    "assume-max-manp",
    llvm::cl::desc(
        "Assume a maximum for the Minimum Arithmetic Noise Padding"));

llvm::cl::list<int64_t> fhelinalgTileSizes(
    "fhelinalg-tile-sizes",
    llvm::cl::desc(
        "Force tiling of FHELinalg operation with the given tile sizes"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);
} // namespace cmdline

llvm::Expected<mlir::concretelang::V0FHEContext> buildFHEContext(
    llvm::Optional<mlir::concretelang::V0FHEConstraint> autoFHEConstraints,
    llvm::Optional<size_t> overrideMaxEintPrecision,
    llvm::Optional<size_t> overrideMaxMANP) {
  if (!autoFHEConstraints.hasValue() &&
      (!overrideMaxMANP.hasValue() || !overrideMaxEintPrecision.hasValue())) {
    return mlir::concretelang::StreamStringError(
        "Maximum encrypted integer precision and maximum for the Minimal"
        "Arithmetic Noise Passing are required, but were neither specified"
        "explicitly nor determined automatically");
  }

  mlir::concretelang::V0FHEConstraint fheConstraints{
      overrideMaxMANP.hasValue() ? overrideMaxMANP.getValue()
                                 : autoFHEConstraints.getValue().norm2,
      overrideMaxEintPrecision.hasValue() ? overrideMaxEintPrecision.getValue()
                                          : autoFHEConstraints.getValue().p};

  const mlir::concretelang::V0Parameter *parameter =
      getV0Parameter(fheConstraints);

  if (!parameter) {
    return mlir::concretelang::StreamStringError()
           << "Could not determine V0 parameters for 2-norm of "
           << fheConstraints.norm2 << " and p of " << fheConstraints.p;
  }

  return mlir::concretelang::V0FHEContext{fheConstraints, *parameter};
}

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

// Process a single source buffer
//
// The parameter `action` specifies how the buffer should be processed
// and thus defines the output.
//
// If the specified action involves JIT compilation, `funcName`
// designates the function to JIT compile. This function is invoked
// using the parameters given in `jitArgs`.
//
// The parameter `parametrizeTFHE` defines, whether the
// parametrization pass for TFHE is executed. If the `action` does
// not involve any MidlFHE manipulation, this parameter does not have
// any effect.
//
// The parameters `overrideMaxEintPrecision` and `overrideMaxMANP`, if
// set, override the values for the maximum required precision of
// encrypted integers and the maximum value for the Minimum Arithmetic
// Noise Padding otherwise determined automatically.
//
// If `verifyDiagnostics` is `true`, the procedure only checks if the
// diagnostic messages provided in the source buffer using
// `expected-error` are produced. If `verifyDiagnostics` is `false`,
// the procedure checks if the parsed module is valid and if all
// requested transformations succeeded.
//
// Compilation output is written to the stream specified by `os`.
mlir::LogicalResult processInputBuffer(
    std::unique_ptr<llvm::MemoryBuffer> buffer, std::string sourceFileName,
    enum Action action, const std::string &funcName,
    llvm::ArrayRef<uint64_t> jitArgs,
    llvm::Optional<size_t> overrideMaxEintPrecision,
    llvm::Optional<size_t> overrideMaxMANP, bool verifyDiagnostics,
    llvm::Optional<llvm::ArrayRef<int64_t>> fhelinalgTileSizes,
    bool autoParallelize, bool loopParallelize, bool dataflowParallelize,
    llvm::Optional<mlir::concretelang::KeySetCache> keySetCache,
    llvm::raw_ostream &os,
    std::shared_ptr<mlir::concretelang::CompilerEngine::Library> outputLib) {
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();

  mlir::concretelang::JitCompilerEngine ce{ccx};

  ce.setVerifyDiagnostics(verifyDiagnostics);
  ce.setAutoParallelize(autoParallelize);
  ce.setLoopParallelize(loopParallelize);
  ce.setDataflowParallelize(dataflowParallelize);
  if (cmdline::passes.size() != 0) {
    ce.setEnablePass([](mlir::Pass *pass) {
      return std::any_of(
          cmdline::passes.begin(), cmdline::passes.end(),
          [&](const std::string &p) { return pass->getArgument() == p; });
    });
  }

  if (overrideMaxEintPrecision.hasValue())
    ce.setMaxEintPrecision(overrideMaxEintPrecision.getValue());

  if (overrideMaxMANP.hasValue())
    ce.setMaxMANP(overrideMaxMANP.getValue());

  ce.setClientParametersFuncName(funcName);
  if (fhelinalgTileSizes.hasValue())
    ce.setFHELinalgTileSizes(*fhelinalgTileSizes);

  if (action == Action::JIT_INVOKE) {
    llvm::Expected<mlir::concretelang::JitCompilerEngine::Lambda> lambdaOrErr =
        ce.buildLambda(std::move(buffer), funcName, keySetCache);

    if (!lambdaOrErr) {
      mlir::concretelang::log_error()
          << "Failed to JIT-compile " << funcName << ": "
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
    case Action::DUMP_BCONCRETE:
      target = mlir::concretelang::CompilerEngine::Target::BCONCRETE;
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

    if (verifyDiagnostics) {
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

  // Convert tile sizes to `Optional`
  llvm::Optional<llvm::ArrayRef<int64_t>> fhelinalgTileSizes;

  if (!cmdline::fhelinalgTileSizes.empty())
    fhelinalgTileSizes.emplace(cmdline::fhelinalgTileSizes);

  llvm::Optional<mlir::concretelang::KeySetCache> jitKeySetCache;
  if (!cmdline::jitKeySetCachePath.empty()) {
    jitKeySetCache =
        mlir::concretelang::KeySetCache(cmdline::jitKeySetCachePath);
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
          std::move(inputBuffer), fileName, cmdline::action, cmdline::funcName,
          cmdline::jitArgs, cmdline::assumeMaxEintPrecision,
          cmdline::assumeMaxMANP, cmdline::verifyDiagnostics,
          fhelinalgTileSizes, cmdline::autoParallelize,
          cmdline::loopParallelize, cmdline::dataflowParallelize,
          jitKeySetCache, os, outputLib);
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
    auto err = outputLib->emitArtifacts();
    if (err) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  if (mlir::failed(compilerMain(argc, argv)))
    return 1;

  return 0;
}
