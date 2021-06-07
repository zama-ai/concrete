#include <iostream>

#include <llvm/Support/CommandLine.h>

#include <memory>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Parser.h>

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
                                  llvm::cl::value_desc("filename"));
}; // namespace cmdline

int main(int argc, char **argv) {
  // Parse command line arguments
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Initialize the MLIR context
  mlir::MLIRContext context;

  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
  context.getOrLoadDialect<mlir::zamalang::MidLFHE::MidLFHEDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();

  // For all input file, parse and dump
  for (const auto &fileName : cmdline::inputs) {
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(fileName, &context);
    if (!module) {
      exit(1);
    }
    module->dump();
  }
  return 0;
}
