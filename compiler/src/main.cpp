#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h"

#include "llvm/Support/SourceMgr.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>


int main(int argc, char **argv) {
  mlir::MLIRContext context;

  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();

  mlir::OpBuilder builder(&context);

  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  module.dump();

  return 0;
}
