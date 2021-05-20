#include <iostream>

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Parser.h>

#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"

int main(int argc, char **argv) {
  mlir::MLIRContext context;

  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();

  if(argc != 2) {
    std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
    return 1;
  }

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);

  if (!module) {
    std::cerr << "Could not parse module" << std::endl;
    return 1;
  }

  module->dump();

  return 0;
}
