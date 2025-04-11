// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <iostream>

#include <concretelang/Conversion/Utils/RTOpConverter.h>
#include <concretelang/Dialect/RT/Analysis/Autopar.h>
#include <concretelang/Dialect/RT/IR/RTDialect.h>
#include <concretelang/Dialect/RT/IR/RTOps.h>
#include <concretelang/Dialect/RT/IR/RTTypes.h>

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include <concretelang/Conversion/Utils/FuncConstOpConversion.h>
#include <concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h>
#include <concretelang/Conversion/Utils/Legality.h>
#include <llvm/IR/Instructions.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/RegionUtils.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/RT/Analysis/Autopar.h.inc>

namespace mlir {
namespace concretelang {

namespace {

class BufferizeRTTypesConverter : public mlir::TypeConverter {
protected:
  bufferization::BufferizeTypeConverter btc;

public:
  BufferizeRTTypesConverter() {
    addConversion([&](mlir::Type type) { return btc.convertType(type); });

    addConversion([&](mlir::RankedTensorType type) {
      return mlir::MemRefType::get(type.getShape(),
                                   this->convertType(type.getElementType()));
    });

    addConversion([&](mlir::UnrankedTensorType type) {
      return mlir::UnrankedMemRefType::get(
          this->convertType(type.getElementType()), 0);
    });

    addConversion([&](mlir::MemRefType type) {
      return mlir::MemRefType::get(type.getShape(),
                                   this->convertType(type.getElementType()),
                                   type.getLayout(), type.getMemorySpace());
    });

    addConversion([&](mlir::UnrankedMemRefType type) {
      return mlir::UnrankedMemRefType::get(
          this->convertType(type.getElementType()), type.getMemorySpace());
    });

    addConversion([&](mlir::concretelang::RT::FutureType type) {
      return mlir::concretelang::RT::FutureType::get(
          this->convertType(type.getElementType()));
    });

    addConversion([&](mlir::concretelang::RT::PointerType type) {
      return mlir::concretelang::RT::PointerType::get(
          this->convertType(type.getElementType()));
    });

    addConversion([&](mlir::FunctionType type) {
      SignatureConversion result(type.getNumInputs());
      mlir::SmallVector<mlir::Type, 1> newResults;

      if (failed(this->convertSignatureArgs(type.getInputs(), result)) ||
          failed(this->convertTypes(type.getResults(), newResults))) {
        return type;
      }

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

    mlir::concretelang::populateWithRTTypeConverterPatterns(patterns, target,
                                                            typeConverter);

    patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
                     mlir::memref::AllocOp, true>,
                 mlir::concretelang::TypeConvertingReinstantiationPattern<
                     mlir::memref::LoadOp>,
                 mlir::concretelang::TypeConvertingReinstantiationPattern<
                     mlir::memref::StoreOp>,
                 mlir::concretelang::TypeConvertingReinstantiationPattern<
                     mlir::memref::CopyOp>,
                 mlir::concretelang::TypeConvertingReinstantiationPattern<
                     mlir::memref::SubViewOp, true>>(&getContext(),
                                                     typeConverter);

    target.addDynamicallyLegalOp<mlir::memref::AllocOp, mlir::memref::LoadOp,
                                 mlir::memref::StoreOp, mlir::memref::CopyOp,
                                 mlir::memref::SubViewOp>(
        [&](mlir::Operation *op) {
          return typeConverter.isLegal(op->getResultTypes()) &&
                 typeConverter.isLegal(op->getOperandTypes());
        });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createBufferizeDataflowTaskOpsPass() {
  return std::make_unique<BufferizeDataflowTaskOpsPass>();
}
} // namespace concretelang
} // namespace mlir
