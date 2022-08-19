// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>

#include <concretelang/Dialect/RT/Analysis/Autopar.h>
#include <concretelang/Dialect/RT/IR/RTDialect.h>
#include <concretelang/Dialect/RT/IR/RTOps.h>
#include <concretelang/Dialect/RT/IR/RTTypes.h>

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include <concretelang/Conversion/Utils/FuncConstOpConversion.h>
#include <concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h>
#include <llvm/IR/Instructions.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/Transforms/RegionUtils.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/RT/Analysis/Autopar.h.inc>

namespace mlir {
namespace concretelang {

namespace {

class BufferizeRTTypesConverter
    : public mlir::bufferization::BufferizeTypeConverter {
public:
  BufferizeRTTypesConverter() {
    addConversion([&](mlir::concretelang::RT::FutureType type) {
      return mlir::concretelang::RT::FutureType::get(
          this->convertType(type.dyn_cast<mlir::concretelang::RT::FutureType>()
                                .getElementType()));
    });
    addConversion([&](mlir::concretelang::RT::PointerType type) {
      return mlir::concretelang::RT::PointerType::get(
          this->convertType(type.dyn_cast<mlir::concretelang::RT::PointerType>()
                                .getElementType()));
    });
    addConversion([&](mlir::FunctionType type) {
      SignatureConversion result(type.getNumInputs());
      mlir::SmallVector<mlir::Type, 1> newResults;
      if (failed(this->convertSignatureArgs(type.getInputs(), result)) ||
          failed(this->convertTypes(type.getResults(), newResults)))
        return type;
      return mlir::FunctionType::get(type.getContext(),
                                     result.getConvertedTypes(), newResults);
    });
  }
};

} // namespace

namespace {
/// For documentation see Autopar.td
struct BufferizeDataflowTaskOpsPass
    : public BufferizeDataflowTaskOpsBase<BufferizeDataflowTaskOpsPass> {

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();
    BufferizeRTTypesConverter typeConverter;

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    patterns.add<FunctionConstantOpConversion<BufferizeRTTypesConverter>>(
        context, typeConverter);

    target.addDynamicallyLegalDialect<mlir::func::FuncDialect>([&](Operation
                                                                       *op) {
      if (auto fun = dyn_cast_or_null<mlir::func::FuncOp>(op))
        return typeConverter.isSignatureLegal(fun.getFunctionType()) &&
               typeConverter.isLegal(&fun.getBody());
      if (auto fun = dyn_cast_or_null<mlir::func::ConstantOp>(op))
        return FunctionConstantOpConversion<BufferizeRTTypesConverter>::isLegal(
            fun, typeConverter);
      return typeConverter.isLegal(op);
    });

    patterns.add<
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::DataflowTaskOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::DataflowYieldOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::MakeReadyFutureOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::AwaitFutureOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::CreateAsyncTaskOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::BuildReturnPtrPlaceholderOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::DerefReturnPtrPlaceholderOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::WorkFunctionReturnOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::RegisterTaskWorkFunctionOp>>(&getContext(),
                                                                 typeConverter);

    // Conversion of RT Dialect Ops
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DataflowTaskOp>(target, typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DataflowYieldOp>(target, typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::MakeReadyFutureOp>(target, typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::AwaitFutureOp>(target, typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::CreateAsyncTaskOp>(target, typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::BuildReturnPtrPlaceholderOp>(target,
                                                             typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>(
        target, typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DerefReturnPtrPlaceholderOp>(target,
                                                             typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::WorkFunctionReturnOp>(target, typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::RegisterTaskWorkFunctionOp>(target,
                                                            typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  BufferizeDataflowTaskOpsPass(bool debug) : debug(debug){};

protected:
  bool debug;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createBufferizeDataflowTaskOpsPass(bool debug) {
  return std::make_unique<BufferizeDataflowTaskOpsPass>(debug);
}
} // namespace concretelang
} // namespace mlir
