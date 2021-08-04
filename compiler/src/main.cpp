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

#include "zamalang/Conversion/Passes.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHETypes.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"
#include "zamalang/Support/CompilerTools.h"

namespace cmdline {

llvm::cl::list<std::string> inputs(llvm::cl::Positional,
                                   llvm::cl::desc("<Input files>"),
                                   llvm::cl::OneOrMore);

llvm::cl::opt<std::string> output("o",
                                  llvm::cl::desc("Specify output filename"),
                                  llvm::cl::value_desc("filename"),
                                  llvm::cl::init("-"));

llvm::cl::opt<bool> verbose("verbose", llvm::cl::desc("verbose logs"),
                            llvm::cl::init<bool>(false));

llvm::cl::list<std::string> passes(
    "passes",
    llvm::cl::desc("Specify the passes to run (use only for compiler tests)"),
    llvm::cl::value_desc("passname"), llvm::cl::ZeroOrMore);

llvm::cl::opt<bool> roundTrip("round-trip",
                              llvm::cl::desc("Just parse and dump"),
                              llvm::cl::init(false));

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

llvm::cl::opt<bool> generateKeySet(
    "generate-keyset",
    llvm::cl::desc("[tmp] Generate a key set for the compiled fhe circuit"),
    llvm::cl::init<bool>(false));

llvm::cl::opt<bool> runJit("run-jit", llvm::cl::desc("JIT the code and run it"),
                           llvm::cl::init<bool>(false));

llvm::cl::opt<std::string> jitFuncname(
    "jit-funcname",
    llvm::cl::desc("Name of the function to execute, default 'main'"),
    llvm::cl::init<std::string>("main"));

llvm::cl::list<int>
    jitArgs("jit-args",
            llvm::cl::desc("Value of arguments to pass to the main func"),
            llvm::cl::value_desc("passname"), llvm::cl::ZeroOrMore);

llvm::cl::opt<bool> toLLVM("to-llvm", llvm::cl::desc("Compile to llvm and "),
                           llvm::cl::init<bool>(false));
}; // namespace cmdline

#define LOG_VERBOSE(expr)                                                      \
  if (cmdline::verbose)                                                        \
    llvm::errs() << expr;

#define LOG_ERROR(expr) llvm::errs() << expr;

auto defaultOptPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);

mlir::LogicalResult dumpLLVMIR(mlir::ModuleOp module, llvm::raw_ostream &os) {
  llvm::LLVMContext context;
  auto llvmModule = mlir::zamalang::CompilerTools::toLLVMModule(
      context, module, defaultOptPipeline);
  if (!llvmModule) {
    return mlir::failure();
  }
  os << **llvmModule;
  return mlir::success();
}

mlir::LogicalResult runJit(mlir::ModuleOp module,
                           mlir::zamalang::KeySet &keySet,
                           llvm::raw_ostream &os) {
  // Create the JIT lambda
  auto maybeLambda = mlir::zamalang::JITLambda::create(
      cmdline::jitFuncname, module, defaultOptPipeline);
  if (!maybeLambda) {
    return mlir::failure();
  }
  auto lambda = std::move(maybeLambda.get());

  // Create the arguments of the JIT lambda
  auto maybeArguments = mlir::zamalang::JITLambda::Argument::create(keySet);
  if (auto err = maybeArguments.takeError()) {

    LOG_ERROR("Cannot create lambda arguments: " << err << "\n");
    return mlir::failure();
  }
  // Set the arguments
  auto arguments = std::move(maybeArguments.get());
  for (auto i = 0; i < cmdline::jitArgs.size(); i++) {
    if (auto err = arguments->setArg(i, cmdline::jitArgs[i])) {
      LOG_ERROR("Cannot push argument " << i << ": " << err << "\n");
      return mlir::failure();
    }
  }
  // Invoke the lambda
  if (lambda->invoke(*arguments)) {
    return mlir::failure();
  }
  uint64_t res = 0;
  if (auto err = arguments->getResult(0, res)) {
    LOG_ERROR("Cannot get result : " << err << "\n");
    return mlir::failure();
  }
  llvm::errs() << res << "\n";
  return mlir::success();
}

// Process a single source buffer
//
// If `verifyDiagnostics` is `true`, the procedure only checks if the
// diagnostic messages provided in the source buffer using
// `expected-error` are produced.
//
// If `verifyDiagnostics` is `false`, the procedure checks if the
// parsed module is valid and if all requested transformations
// succeeded.
mlir::LogicalResult
processInputBuffer(mlir::MLIRContext &context,
                   std::unique_ptr<llvm::MemoryBuffer> buffer,
                   llvm::raw_ostream &os, bool verifyDiagnostics) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  mlir::SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr,
                                                            &context);
  auto module = mlir::parseSourceFile(sourceMgr, &context);

  if (verifyDiagnostics)
    return sourceMgrHandler.verify();

  if (!module)
    return mlir::failure();

  if (cmdline::roundTrip) {
    module->print(os);
    return mlir::success();
  }

  auto enablePass = [](std::string passName) {
    return cmdline::passes.size() == 0 ||
           std::any_of(cmdline::passes.begin(), cmdline::passes.end(),
                       [&](const std::string &p) { return passName == p; });
  };

  // Lower to MLIR Stds Dialects and compute the constraint on the FHE Circuit.
  mlir::zamalang::FHECircuitConstraint constraint;
  LOG_VERBOSE("### Lower from HLFHE to MLIR standards \n");
  if (mlir::zamalang::CompilerTools::lowerHLFHEToMlirStdsDialect(
          context, *module, constraint, enablePass)
          .failed()) {
    return mlir::failure();
  }
  LOG_VERBOSE("### Global FHE constraint: {norm2:" << constraint.norm2 << ", p:"
                                                   << constraint.p << "}\n");

  // Retreive the parameters for the v0 approach
  mlir::zamalang::V0Parameter *fheParameter =
      mlir::zamalang::getV0Parameter(constraint.norm2, constraint.p);
  LOG_VERBOSE("### FHE parameters for the atomic pattern: {k: "
              << fheParameter->k
              << ", polynomialSize: " << fheParameter->polynomialSize
              << ", nSmall: " << fheParameter->nSmall
              << ", brLevel: " << fheParameter->brLevel
              << ", brLogBase: " << fheParameter->brLogBase
              << ", ksLevel: " << fheParameter->ksLevel
              << ", polynomialSize: " << fheParameter->ksLogBase << "}\n");

  // Generate the keySet
  std::unique_ptr<mlir::zamalang::KeySet> keySet;
  if (cmdline::generateKeySet || cmdline::runJit) {
    // Create the client parameters
    auto clientParameter = mlir::zamalang::createClientParametersForV0(
        fheParameter, constraint.p, cmdline::jitFuncname, *module);
    if (auto err = clientParameter.takeError()) {
      LOG_ERROR("cannot generate client parameters: " << err << "\n");
      return mlir::failure();
    }
    LOG_VERBOSE("### Generate the key set\n");
    auto maybeKeySet =
        mlir::zamalang::KeySet::generate(clientParameter.get(), 0,
                                         0); // TODO: seed
    if (auto err = maybeKeySet.takeError()) {
      llvm::errs() << err;
      return mlir::failure();
    }
    keySet = std::move(maybeKeySet.get());
  }

  // Lower to MLIR LLVM Dialect
  LOG_VERBOSE("### Lower from MLIR standards to LLVM\n");
  if (mlir::zamalang::CompilerTools::lowerMlirStdsDialectToMlirLLVMDialect(
          context, *module, enablePass)
          .failed()) {
    return mlir::failure();
  }

  if (cmdline::runJit) {
    LOG_VERBOSE("### JIT compile & running\n");
    return runJit(module.get(), *keySet, os);
  }
  if (cmdline::toLLVM) {
    return dumpLLVMIR(module.get(), os);
  }
  module->print(os);
  return mlir::success();
}

mlir::LogicalResult compilerMain(int argc, char **argv) {
  // Parse command line arguments
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Initialize the MLIR context
  mlir::MLIRContext context;

  // String for error messages from library functions
  std::string errorMessage;

  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
  context.getOrLoadDialect<mlir::zamalang::MidLFHE::MidLFHEDialect>();
  context.getOrLoadDialect<mlir::zamalang::LowLFHE::LowLFHEDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();

  if (cmdline::verifyDiagnostics)
    context.printOpOnDiagnostic(false);

  auto output = mlir::openOutputFile(cmdline::output, &errorMessage);

  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }

  // Iterate over all inpiut files specified on the command line
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
                return processInputBuffer(context, std::move(inputBuffer), os,
                                          cmdline::verifyDiagnostics);
              },
              output->os())))
        return mlir::failure();
    } else {
      return processInputBuffer(context, std::move(file), output->os(),
                                cmdline::verifyDiagnostics);
    }
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  if (mlir::failed(compilerMain(argc, argv)))
    return 1;

  return 0;
}
