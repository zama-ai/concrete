#include <iostream>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "zamalang/Conversion/HLFHEToMidLFHE/Patterns.h"
#include "zamalang/Conversion/Passes.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

namespace {
struct HLFHEToMidLFHEPass : public HLFHEToMidLFHEBase<HLFHEToMidLFHEPass> {
  void runOnOperation() final;
};
} // namespace

using mlir::zamalang::HLFHE::EncryptedIntegerType;
using mlir::zamalang::MidLFHE::GLWECipherTextType;

/// HLFHEToMidLFHETypeConverter is a TypeConverter that transform
/// `HLFHE.eint<p>` to `MidLFHE.gwle<{_,_,_}{p}>`
class HLFHEToMidLFHETypeConverter : public mlir::TypeConverter {

public:
  HLFHEToMidLFHETypeConverter() {
    addConversion([&](EncryptedIntegerType type) {
      return mlir::zamalang::convertTypeEncryptedIntegerToGLWE(
          type.getContext(), type);
    });
    addConversion([&](mlir::MemRefType type) {
      auto eint =
          type.getElementType().dyn_cast_or_null<EncryptedIntegerType>();
      if (eint == nullptr) {
        return (mlir::Type)(type);
      }
      mlir::Type r = mlir::MemRefType::get(
          type.getShape(),
          mlir::zamalang::convertTypeEncryptedIntegerToGLWE(eint.getContext(),
                                                            eint),
          type.getAffineMaps(), type.getMemorySpace());
      return r;
    });
  }

  /// [workaround] as converter.isLegal returns unexpected false for glwe with
  /// same parameters.
  static bool _isLegal(mlir::Type type) {
    if (type.isa<EncryptedIntegerType>()) {
      return false;
    }
    auto memref = type.dyn_cast_or_null<mlir::MemRefType>();
    if (memref != nullptr) {
      return _isLegal(memref.getElementType());
    }
    return true;
  }

  // [workaround]
  template <typename TypeRangeT> static bool _isLegal(TypeRangeT &&types) {
    return llvm::all_of(types,
                        [&](const mlir::Type ty) { return _isLegal(ty); });
  }
};

/// LinalgGenericPattern is a rewrite pattern that convert types from HLFHE to
/// MidLFHE of `linalg.generic` operation
struct LinalgGenericPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  LinalgGenericPattern(mlir::MLIRContext *context,
                       mlir::PatternBenefit benefit = 100)
      : mlir::OpRewritePattern<mlir::linalg::GenericOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {

    HLFHEToMidLFHETypeConverter converter;

    rewriter.startRootUpdate(op);
    // Rewrite arguments
    {
      for (auto i = 0; i < op->getNumOperands(); i++) {
        auto operand = op->getOperand(i);
        mlir::Type type = converter.convertType(operand.getType());
        if (type != mlir::Type()) {
          operand.setType(type);
        }
      }
    }
    // Rewrite results
    {
      for (auto i = 0; i < op->getNumResults(); i++) {
        auto result = op->getResult(i);
        mlir::Type type = converter.convertType(result.getType());
        if (type != mlir::Type()) {
          result.setType(type);
        }
      }
    }
    // Rewrite block arguments
    mlir::Region &region = op->getRegion(0);
    mlir::Block *entry = &region.front();
    for (auto arg : entry->getArguments()) {
      mlir::Type type = converter.convertType(arg.getType());
      if (type != mlir::Type()) {
        arg.setType(type);
      }
    }
    rewriter.finalizeRootUpdate(op);
    return mlir::success();
  }
};

void HLFHEToMidLFHEPass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  HLFHEToMidLFHETypeConverter converter;

  // Mark ops from the target dialect as legal operations
  target.addLegalDialect<mlir::zamalang::MidLFHE::MidLFHEDialect>();

  // Make sure that no ops from `HLFHE` remain after the lowering
  target.addIllegalDialect<mlir::zamalang::HLFHE::HLFHEDialect>();

  // Make sure that no ops `linalg.generic` that have illegal types
  target.addDynamicallyLegalOp<mlir::linalg::GenericOp>(
      [&](mlir::linalg::GenericOp op) {
        return (converter.isLegal(op.getOperandTypes()) &&
                converter.isLegal(op.getResultTypes()) &&
                converter.isLegal(op->getRegion(0).front().getArgumentTypes()));
      });

  // Make sure that func has legal signature
  target.addDynamicallyLegalOp<mlir::FuncOp>([](mlir::FuncOp funcOp) {
    HLFHEToMidLFHETypeConverter converter;
    // [workaround] should be this commented code but for an unknown reasons
    // converter.isLegal returns false for glwe with same parameters.
    //
    // return converter.isSignatureLegal(op.getType()) &&
    //  converter.isLegal(&op.getBody());
    auto funcType = funcOp.getType();
    return HLFHEToMidLFHETypeConverter::_isLegal(funcType.getInputs()) &&
           HLFHEToMidLFHETypeConverter::_isLegal(funcType.getResults());
  });
  // Add all patterns required to lower all ops from `HLFHE` to
  // `MidLFHE`
  mlir::OwningRewritePatternList patterns(&getContext());

  populateWithGenerated(patterns);
  patterns.add<LinalgGenericPattern>(&getContext());
  mlir::populateFuncOpTypeConversionPattern(patterns, converter);

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

namespace mlir {
namespace zamalang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertHLFHEToMidLFHEPass() {
  return std::make_unique<HLFHEToMidLFHEPass>();
}
} // namespace zamalang
} // namespace mlir
