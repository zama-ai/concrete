// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "concretelang/Conversion/Tools.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"

mlir::LogicalResult insertForwardDeclaration(mlir::Operation *op,
                                             mlir::OpBuilder &rewriter,
                                             llvm::StringRef funcName,
                                             mlir::FunctionType funcType) {
  // Looking for the `funcName` Operation
  auto module = mlir::SymbolTable::getNearestSymbolTable(op);
  auto opFunc = mlir::dyn_cast_or_null<mlir::SymbolOpInterface>(
      mlir::SymbolTable::lookupSymbolIn(module, funcName));
  if (!opFunc) {
    // Insert the forward declaration of the funcName
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&module->getRegion(0).front());

    opFunc = rewriter.create<mlir::func::FuncOp>(rewriter.getUnknownLoc(),
                                                 funcName, funcType);
    opFunc.setPrivate();
  } else {
    // Check if the `funcName` is well a private function
    if (!opFunc.isPrivate()) {
      op->emitError() << "the function \"" << funcName
                      << "\" conflicts with the concrete C API, please rename";
      return mlir::failure();
    }
  }
  assert(llvm::isa<mlir::FunctionOpInterface>(
      mlir::SymbolTable::lookupSymbolIn(module, funcName)));
  return mlir::success();
}

/// Returns the value of the context argument from the enclosing func
mlir::Value getContextArgument(mlir::Operation *op) {
  mlir::Block *block = op->getBlock();
  while (block != nullptr) {
    if (llvm::isa<mlir::func::FuncOp>(block->getParentOp())) {
      block = &mlir::cast<mlir::func::FuncOp>(block->getParentOp())
                   .getBody()
                   .front();

      auto context = std::find_if(
          block->getArguments().rbegin(), block->getArguments().rend(),
          [](mlir::BlockArgument &arg) {
            return arg.getType()
                .isa<mlir::concretelang::Concrete::ContextType>();
          });

      assert(context != block->getArguments().rend() &&
             "Cannot find the Concrete.context");

      return *context;
    }
    block = block->getParentOp()->getBlock();
  }
  assert(false); // can't find a function that enclose the op
  return nullptr;
}
