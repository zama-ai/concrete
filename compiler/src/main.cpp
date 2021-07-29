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

llvm::cl::opt<bool> runJit("run-jit", llvm::cl::desc("JIT the code and run it"),
                           llvm::cl::init<bool>(false));
llvm::cl::list<int>
    jitArgs("jit-args",
            llvm::cl::desc("Value of arguments to pass to the main func"),
            llvm::cl::value_desc("passname"), llvm::cl::ZeroOrMore);

llvm::cl::opt<bool> toLLVM("to-llvm", llvm::cl::desc("Compile to llvm and "),
                           llvm::cl::init<bool>(false));
}; // namespace cmdline

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

mlir::LogicalResult runJit(mlir::ModuleOp module, llvm::raw_ostream &os) {
  // Create the JIT lambda
  auto maybeLambda =
      mlir::zamalang::JITLambda::create("main", module, defaultOptPipeline);
  if (!maybeLambda) {
    return mlir::failure();
  }
  auto lambda = maybeLambda.get().get();

  // Create buffer to copy argument
  std::vector<int64_t> dummy(cmdline::jitArgs.size());
  llvm::SmallVector<void *> llvmArgs;
  for (auto i = 0; i < cmdline::jitArgs.size(); i++) {
    dummy[i] = cmdline::jitArgs[i];
    llvmArgs.push_back(&dummy[i]);
  }
  // Add the result pointer
  uint64_t res = 0;
  llvmArgs.push_back(&res);

  // Invoke the lambda
  if (lambda->invokeRaw(llvmArgs)) {
    return mlir::failure();
  }

  std::cerr << res << "\n";
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

  if (mlir::zamalang::CompilerTools::lowerHLFHEToMlirLLVMDialect(
          context, *module,
          [](std::string passName) {
            return cmdline::passes.size() == 0 ||
                   std::any_of(
                       cmdline::passes.begin(), cmdline::passes.end(),
                       [&](const std::string &p) { return passName == p; });
          })
          .failed()) {
    return mlir::failure();
  }

  if (cmdline::runJit) {
    return runJit(module.get(), os);
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
