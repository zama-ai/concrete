#include <iostream>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Parser.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/ToolUtilities.h>

#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

namespace cmdline {
llvm::cl::list<std::string> inputs(llvm::cl::Positional,
                                   llvm::cl::desc("<Input files>"),
                                   llvm::cl::OneOrMore);

llvm::cl::opt<std::string> output("o",
                                  llvm::cl::desc("Specify output filename"),
                                  llvm::cl::value_desc("filename"),
                                  llvm::cl::init("-"));

llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));
}; // namespace cmdline

// Process a single source buffer
mlir::LogicalResult
processInputBuffer(mlir::MLIRContext &context,
                   std::unique_ptr<llvm::MemoryBuffer> buffer,
                   llvm::raw_ostream &os) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  auto module = mlir::parseSourceFile(sourceMgr, &context);

  if (!module)
    return mlir::failure();

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
                return processInputBuffer(context, std::move(inputBuffer), os);
              },
              output->os())))
        return mlir::failure();
    } else {
      return processInputBuffer(context, std::move(file), output->os());
    }
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  if (mlir::failed(compilerMain(argc, argv)))
    return 1;

  return 0;
}
