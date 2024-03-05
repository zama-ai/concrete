// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Operation.h>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

template <typename TypeConverterType>
struct FunctionConstantOpConversion
    : public mlir::OpRewritePattern<mlir::func::ConstantOp> {
  FunctionConstantOpConversion(mlir::MLIRContext *ctx,
                               TypeConverterType &converter,
                               mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::func::ConstantOp>(ctx, benefit),
        converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto symTab = mlir::SymbolTable::getNearestSymbolTable(op);
    auto funcOp = mlir::SymbolTable::lookupSymbolIn(symTab, op.getValue());
    assert(funcOp &&
           "Function symbol missing in symbol table for function constant op.");
    mlir::FunctionType funType = mlir::cast<mlir::func::FuncOp>(funcOp)
                                     .getFunctionType()
                                     .cast<mlir::FunctionType>();
    typename TypeConverterType::SignatureConversion result(
        funType.getNumInputs());
    mlir::SmallVector<mlir::Type, 1> newResults;
    if (failed(converter.convertSignatureArgs(funType.getInputs(), result)) ||
        failed(converter.convertTypes(funType.getResults(), newResults)))
      return mlir::failure();
    auto newType = mlir::FunctionType::get(
        rewriter.getContext(), result.getConvertedTypes(), newResults);
    rewriter.updateRootInPlace(op, [&] { op.getResult().setType(newType); });
    return mlir::success();
  }

  static bool isLegal(mlir::func::ConstantOp fun,
                      TypeConverterType &converter) {
    auto symTab = mlir::SymbolTable::getNearestSymbolTable(fun);
    auto funcOp = mlir::SymbolTable::lookupSymbolIn(symTab, fun.getValue());
    assert(funcOp &&
           "Function symbol missing in symbol table for function constant op.");
    mlir::FunctionType funType = mlir::cast<mlir::func::FuncOp>(funcOp)
                                     .getFunctionType()
                                     .cast<mlir::FunctionType>();
    typename TypeConverterType::SignatureConversion result(
        funType.getNumInputs());
    mlir::SmallVector<mlir::Type, 1> newResults;
    if (failed(converter.convertSignatureArgs(funType.getInputs(), result)) ||
        failed(converter.convertTypes(funType.getResults(), newResults)))
      return false;
    auto newType = mlir::FunctionType::get(
        fun.getContext(), result.getConvertedTypes(), newResults);
    return newType == fun.getType();
  }

private:
  TypeConverterType &converter;
};
