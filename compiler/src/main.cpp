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

#include "mlir/IR/BuiltinOps.h"
#include "zamalang/Conversion/Passes.h"
#include "zamalang/Conversion/Utils/GlobalFHEContext.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHETypes.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"
#include "zamalang/Support/Jit.h"
#include "zamalang/Support/KeySet.h"
#include "zamalang/Support/Pipeline.h"
#include "zamalang/Support/logging.h"

enum EntryDialect { HLFHE, MIDLFHE, LOWLFHE, STD, LLVM };

enum Action {
  ROUND_TRIP,
  DUMP_HLFHE_MANP,
  DUMP_MIDLFHE,
  DUMP_LOWLFHE,
  DUMP_STD,
  DUMP_LLVM_DIALECT,
  DUMP_LLVM_IR,
  DUMP_OPTIMIZED_LLVM_IR,
  JIT_INVOKE
};

namespace cmdline {
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
                                  llvm::cl::init("-"));

llvm::cl::opt<bool> verbose("verbose", llvm::cl::desc("verbose logs"),
                            llvm::cl::init<bool>(false));

llvm::cl::opt<bool> parametrizeMidLFHE(
    "parametrize-midlfhe",
    llvm::cl::desc("Perform MidLFHE global parametrization pass"),
    llvm::cl::init<bool>(true));

static llvm::cl::opt<enum EntryDialect> entryDialect(
    "e", "entry-dialect", llvm::cl::desc("Entry dialect"),
    llvm::cl::init<enum EntryDialect>(EntryDialect::HLFHE),
    llvm::cl::ValueRequired, llvm::cl::NumOccurrencesFlag::Required,
    llvm::cl::values(
        clEnumValN(EntryDialect::HLFHE, "hlfhe",
                   "Input module is composed of HLFHE operations")),
    llvm::cl::values(
        clEnumValN(EntryDialect::MIDLFHE, "midlfhe",
                   "Input module is composed of MidLFHE operations")),
    llvm::cl::values(
        clEnumValN(EntryDialect::LOWLFHE, "lowlfhe",
                   "Input module is composed of LowLFHE operations")),
    llvm::cl::values(
        clEnumValN(EntryDialect::STD, "std",
                   "Input module is composed of operations from std")),
    llvm::cl::values(
        clEnumValN(EntryDialect::LLVM, "llvm",
                   "Input module is composed of operations from llvm")));

static llvm::cl::opt<enum Action> action(
    "a", "action", llvm::cl::desc("output mode"), llvm::cl::ValueRequired,
    llvm::cl::NumOccurrencesFlag::Required,
    llvm::cl::values(
        clEnumValN(Action::ROUND_TRIP, "roundtrip",
                   "Parse input module and regenerate textual representation")),
    llvm::cl::values(clEnumValN(Action::DUMP_HLFHE_MANP, "dump-hlfhe-manp",
                                "Dump HLFHE module after running the Minimal "
                                "Arithmetic Noise Padding pass")),
    llvm::cl::values(clEnumValN(Action::DUMP_MIDLFHE, "dump-midlfhe",
                                "Lower to MidLFHE and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_LOWLFHE, "dump-lowlfhe",
                                "Lower to LowLFHE and dump result")),
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
                                "function specified with --jit-funcname")));

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

llvm::cl::opt<std::string> jitFuncName(
    "jit-funcname",
    llvm::cl::desc("Name of the function to execute, default 'main'"),
    llvm::cl::init<std::string>("main"));

llvm::cl::list<uint64_t>
    jitArgs("jit-args",
            llvm::cl::desc("Value of arguments to pass to the main func"),
            llvm::cl::value_desc("argument(uint64)"), llvm::cl::ZeroOrMore);

llvm::cl::opt<llvm::Optional<size_t>, false, OptionalSizeTParser>
    assumeMaxEintPrecision(
        "assume-max-eint-precision",
        llvm::cl::desc("Assume a maximum precision for encrypted integers"));

llvm::cl::opt<llvm::Optional<size_t>, false, OptionalSizeTParser> assumeMaxMANP(
    "assume-max-manp",
    llvm::cl::desc(
        "Assume a maximum for the Minimum Arithmetic Noise Padding"));

}; // namespace cmdline

std::function<llvm::Error(llvm::Module *)> defaultOptPipeline =
    mlir::makeOptimizingTransformer(3, 0, nullptr);

std::unique_ptr<mlir::zamalang::KeySet>
generateKeySet(mlir::ModuleOp &module, mlir::zamalang::V0FHEContext &fheContext,
               const std::string &jitFuncName) {
  std::unique_ptr<mlir::zamalang::KeySet> keySet;

  mlir::zamalang::log_verbose()
      << "### Global FHE constraint: {norm2:" << fheContext.constraint.norm2
      << ", p:" << fheContext.constraint.p << "}\n";
  mlir::zamalang::log_verbose()
      << "### FHE parameters for the atomic pattern: {k: "
      << fheContext.parameter.k
      << ", polynomialSize: " << fheContext.parameter.polynomialSize
      << ", nSmall: " << fheContext.parameter.nSmall
      << ", brLevel: " << fheContext.parameter.brLevel
      << ", brLogBase: " << fheContext.parameter.brLogBase
      << ", ksLevel: " << fheContext.parameter.ksLevel
      << ", ksLogBase: " << fheContext.parameter.ksLogBase << "}\n";

  // Create the client parameters
  auto clientParameter = mlir::zamalang::createClientParametersForV0(
      fheContext, jitFuncName, module);

  if (auto err = clientParameter.takeError()) {
    mlir::zamalang::log_error()
        << "cannot generate client parameters: " << err << "\n";
    return nullptr;
  }

  mlir::zamalang::log_verbose() << "### Generate the key set\n";

  auto maybeKeySet = mlir::zamalang::KeySet::generate(clientParameter.get(), 0,
                                                      0); // TODO: seed
  if (auto err = maybeKeySet.takeError()) {
    llvm::errs() << err;
    return nullptr;
  }

  return std::move(maybeKeySet.get());
}

llvm::Expected<mlir::zamalang::V0FHEContext> buildFHEContext(
    llvm::Optional<mlir::zamalang::V0FHEConstraint> autoFHEConstraints,
    llvm::Optional<size_t> overrideMaxEintPrecision,
    llvm::Optional<size_t> overrideMaxMANP) {
  if (!autoFHEConstraints.hasValue() &&
      (!overrideMaxMANP.hasValue() || !overrideMaxEintPrecision.hasValue())) {
    return llvm::make_error<llvm::StringError>(
        "Maximum encrypted integer precision and maximum for the Minimal"
        "Arithmetic Noise Passing are required, but were neither specified"
        "explicitly nor determined automatically",
        llvm::inconvertibleErrorCode());
  }

  mlir::zamalang::V0FHEConstraint fheConstraints{
      .norm2 = overrideMaxMANP.hasValue() ? overrideMaxMANP.getValue()
                                          : autoFHEConstraints.getValue().norm2,
      .p = overrideMaxEintPrecision.hasValue()
               ? overrideMaxEintPrecision.getValue()
               : autoFHEConstraints.getValue().p};

  const mlir::zamalang::V0Parameter *parameter = getV0Parameter(fheConstraints);

  if (!parameter) {
    std::string buffer;
    llvm::raw_string_ostream strs(buffer);
    strs << "Could not determine V0 parameters for 2-norm of "
         << fheConstraints.norm2 << " and p of " << fheConstraints.p;

    return llvm::make_error<llvm::StringError>(strs.str(),
                                               llvm::inconvertibleErrorCode());
  }

  return mlir::zamalang::V0FHEContext{fheConstraints, *parameter};
}

mlir::LogicalResult buildAssignFHEContext(
    llvm::Optional<mlir::zamalang::V0FHEContext> &fheContext,
    llvm::Optional<mlir::zamalang::V0FHEConstraint> autoFHEConstraints,
    llvm::Optional<size_t> overrideMaxEintPrecision,
    llvm::Optional<size_t> overrideMaxMANP) {

  if (fheContext.hasValue())
    return mlir::success();

  llvm::Expected<mlir::zamalang::V0FHEContext> fheContextOrErr =
      buildFHEContext(autoFHEConstraints, overrideMaxEintPrecision,
                      overrideMaxMANP);

  if (auto err = fheContextOrErr.takeError()) {
    mlir::zamalang::log_error() << err;
    return mlir::failure();
  }

  fheContext.emplace(fheContextOrErr.get());

  return mlir::success();
}

// Process a single source buffer
//
// The parameter `entryDialect` must specify the FHE dialect to which
// belong all FHE operations used in the source buffer. The input
// program must only contain FHE operations from that single FHE
// dialect, otherwise processing might fail.
//
// The parameter `action` specifies how the buffer should be processed
// and thus defines the output.
//
// If the specified action involves JIT compilation, `jitFuncName`
// designates the function to JIT compile. This function is invoked
// using the parameters given in `jitArgs`.
//
// The parameter `parametrizeMidLFHE` defines, whether the
// parametrization pass for MidLFHE is executed. If the pair of
// `entryDialect` and `action` does not involve any MidlFHE
// manipulation, this parameter does not have any effect.
//
// The parameters `overrideMaxEintPrecision` and `overrideMaxMANP`, if
// set, override the values for the maximum required precision of
// encrypted integers and the maximum value for the Minimum Arithmetic
// Noise Padding otherwise determined automatically if the entry
// dialect is HLFHE..
//
// If `verifyDiagnostics` is `true`, the procedure only checks if the
// diagnostic messages provided in the source buffer using
// `expected-error` are produced. If `verifyDiagnostics` is `false`,
// the procedure checks if the parsed module is valid and if all
// requested transformations succeeded.
//
// If `verbose` is true, debug messages are displayed throughout the
// compilation process.
//
// Compilation output is written to the stream specified by `os`.
mlir::LogicalResult processInputBuffer(
    mlir::MLIRContext &context, std::unique_ptr<llvm::MemoryBuffer> buffer,
    enum EntryDialect entryDialect, enum Action action,
    const std::string &jitFuncName, llvm::ArrayRef<uint64_t> jitArgs,
    bool parametrizeMidlHFE, llvm::Optional<size_t> overrideMaxEintPrecision,
    llvm::Optional<size_t> overrideMaxMANP, bool verifyDiagnostics,
    bool verbose, llvm::raw_ostream &os) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  mlir::SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr,
                                                            &context);
  mlir::OwningModuleRef moduleRef = mlir::parseSourceFile(sourceMgr, &context);

  llvm::Optional<mlir::zamalang::V0FHEConstraint> fheConstraints;
  llvm::Optional<mlir::zamalang::V0FHEContext> fheContext;

  std::unique_ptr<mlir::zamalang::KeySet> keySet = nullptr;

  if (verbose)
    context.disableMultithreading();

  if (verifyDiagnostics)
    return sourceMgrHandler.verify();

  if (!moduleRef)
    return mlir::failure();

  mlir::ModuleOp module = moduleRef.get();

  if (action == Action::ROUND_TRIP) {
    module->print(os);
    return mlir::success();
  }

  // Lowering pipeline. Each stage is represented as a label in the
  // switch statement, from the most abstract dialect to the lowest
  // level. Every labels acts as an entry point into the pipeline with
  // a fallthrough mechanism to the next stage. Actions act as exit
  // points from the pipeline.
  switch (entryDialect) {
  case EntryDialect::HLFHE:
    if (action == Action::DUMP_HLFHE_MANP) {
      if (mlir::zamalang::pipeline::invokeMANPPass(context, module, false)
              .failed()) {
        return mlir::failure();
      }

      module.print(os);
      return mlir::success();
    } else {
      llvm::Expected<llvm::Optional<mlir::zamalang::V0FHEConstraint>>
          fheConstraintsOrErr =
              mlir::zamalang::pipeline::getFHEConstraintsFromHLFHE(context,
                                                                   module);
      if (auto err = fheConstraintsOrErr.takeError()) {
        mlir::zamalang::log_error() << err;
        return mlir::failure();
      } else {
        fheConstraints = fheConstraintsOrErr.get();
      }
    }

    if (mlir::zamalang::pipeline::lowerHLFHEToMidLFHE(context, module, verbose)
            .failed())
      return mlir::failure();

    // fallthrough
  case EntryDialect::MIDLFHE:
    if (action == Action::DUMP_MIDLFHE) {
      module.print(os);
      return mlir::success();
    }

    if (buildAssignFHEContext(fheContext, fheConstraints,
                              overrideMaxEintPrecision, overrideMaxMANP)
            .failed()) {
      return mlir::failure();
    }

    if (mlir::zamalang::pipeline::lowerMidLFHEToLowLFHE(
            context, module, fheContext.getValue(), parametrizeMidlHFE)
            .failed())
      return mlir::failure();

    // fallthrough
  case EntryDialect::LOWLFHE:
    if (action == Action::DUMP_LOWLFHE) {
      module.print(os);
      return mlir::success();
    }

    if (mlir::zamalang::pipeline::lowerLowLFHEToStd(context, module).failed())
      return mlir::failure();

    // fallthrough
  case EntryDialect::STD:
    if (action == Action::DUMP_STD) {
      module.print(os);
      return mlir::success();
    } else if (action == Action::JIT_INVOKE) {
      if (buildAssignFHEContext(fheContext, fheConstraints,
                                overrideMaxEintPrecision, overrideMaxMANP)
              .failed()) {
        return mlir::failure();
      }

      keySet = generateKeySet(module, fheContext.getValue(), jitFuncName);
    }

    if (mlir::zamalang::pipeline::lowerStdToLLVMDialect(context, module,
                                                        verbose)
            .failed())
      return mlir::failure();

    // fallthrough
  case EntryDialect::LLVM: {
    if (action == Action::DUMP_LLVM_DIALECT) {
      module.print(os);
      return mlir::success();
    } else if (action == Action::JIT_INVOKE) {
      return mlir::zamalang::runJit(module, jitFuncName, jitArgs, *keySet,
                                    defaultOptPipeline, os);
    }

    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule =
        mlir::zamalang::pipeline::lowerLLVMDialectToLLVMIR(context, llvmContext,
                                                           module);

    if (!llvmModule) {
      mlir::zamalang::log_error()
          << "Failed to translate LLVM dialect to LLVM IR\n";
      return mlir::failure();
    }

    if (action == Action::DUMP_LLVM_IR) {
      llvmModule->dump();
      return mlir::success();
    }

    if (mlir::zamalang::pipeline::optimizeLLVMModule(llvmContext, *llvmModule)
            .failed()) {
      mlir::zamalang::log_error() << "Failed to optimize LLVM IR\n";
      return mlir::failure();
    }

    if (action == Action::DUMP_OPTIMIZED_LLVM_IR) {
      llvmModule->dump();
      return mlir::success();
    }

    break;
  }
  }

  return mlir::success();
}

mlir::LogicalResult compilerMain(int argc, char **argv) {
  // Parse command line arguments
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Initialize the MLIR context
  mlir::MLIRContext context;

  mlir::zamalang::setupLogging(cmdline::verbose);

  // String for error messages from library functions
  std::string errorMessage;

  if (cmdline::action == Action::DUMP_HLFHE_MANP &&
      cmdline::entryDialect != EntryDialect::HLFHE) {
    mlir::zamalang::log_error()
        << "Can only invoke Minimal Arithmetic Noise pass on HLFHE programs";
    return mlir::failure();
  }

  if (cmdline::action == Action::JIT_INVOKE &&
      cmdline::entryDialect != EntryDialect::HLFHE &&
      cmdline::entryDialect != EntryDialect::MIDLFHE &&
      cmdline::entryDialect != EntryDialect::LOWLFHE &&
      cmdline::entryDialect != EntryDialect::STD) {
    mlir::zamalang::log_error()
        << "Can only JIT invoke HLFHE / MidLFHE / LowLFHE / STD programs";
    return mlir::failure();
  }

  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
  context.getOrLoadDialect<mlir::zamalang::MidLFHE::MidLFHEDialect>();
  context.getOrLoadDialect<mlir::zamalang::LowLFHE::LowLFHEDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  if (cmdline::verifyDiagnostics)
    context.printOpOnDiagnostic(false);

  auto output = mlir::openOutputFile(cmdline::output, &errorMessage);

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
    if (cmdline::splitInputFile) {
      if (mlir::failed(mlir::splitAndProcessBuffer(
              std::move(file),
              [&](std::unique_ptr<llvm::MemoryBuffer> inputBuffer,
                  llvm::raw_ostream &os) {
                return processInputBuffer(
                    context, std::move(inputBuffer), cmdline::entryDialect,
                    cmdline::action, cmdline::jitFuncName, cmdline::jitArgs,
                    cmdline::parametrizeMidLFHE,
                    cmdline::assumeMaxEintPrecision, cmdline::assumeMaxMANP,
                    cmdline::verifyDiagnostics, cmdline::verbose, os);
              },
              output->os())))
        return mlir::failure();
    } else {
      return processInputBuffer(
          context, std::move(file), cmdline::entryDialect, cmdline::action,
          cmdline::jitFuncName, cmdline::jitArgs, cmdline::parametrizeMidLFHE,
          cmdline::assumeMaxEintPrecision, cmdline::assumeMaxMANP,
          cmdline::verifyDiagnostics, cmdline::verbose, output->os());
    }
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  if (mlir::failed(compilerMain(argc, argv)))
    return 1;

  return 0;
}
