#include "mlir//IR/BuiltinTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "zamalang/Conversion/Passes.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEOps.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHETypes.h"

class LowLFHEToConcreteCAPITypeConverter : public mlir::TypeConverter {

public:
  LowLFHEToConcreteCAPITypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](mlir::zamalang::LowLFHE::PlaintextType type) {
      return mlir::IntegerType::get(type.getContext(), 64);
    });
    addConversion([&](mlir::zamalang::LowLFHE::CleartextType type) {
      return mlir::IntegerType::get(type.getContext(), 64);
    });
  }
};

mlir::LogicalResult insertForwardDeclaration(mlir::Operation *op,
                                             mlir::PatternRewriter &rewriter,
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

    opFunc = rewriter.create<mlir::FuncOp>(rewriter.getUnknownLoc(), funcName,
                                           funcType);
    opFunc.setPrivate();
  } else {
    // Check if the `funcName` is well a private function
    if (!opFunc.isPrivate()) {
      op->emitError() << "the function \"" << funcName
                      << "\" conflicts with the concrete C API, please rename";
      return mlir::failure();
    }
  }
  assert(mlir::SymbolTable::lookupSymbolIn(module, funcName)
             ->template hasTrait<mlir::OpTrait::FunctionLike>());
  return mlir::success();
}

/// LowLFHEOpToConcreteCAPICallPattern<Op> match the `Op` Operation and
/// replace with a call to `funcName`, the funcName should be an external
/// function that was linked later. It insert the forward declaration of the
/// private `funcName` if it not already in the symbol table.
/// The C signature of the function should be `void funcName(int *err, out,
/// arg0, arg1)`, the pattern rewrite:
/// ```
/// out = op(arg0, arg1)
/// ```
/// to
/// ```
/// err = constant 0 : i64
/// call_op(err, out, arg0, arg1);
/// ```
template <typename Op>
struct LowLFHEOpToConcreteCAPICallPattern : public mlir::OpRewritePattern<Op> {
  LowLFHEOpToConcreteCAPICallPattern(mlir::MLIRContext *context,
                                     mlir::StringRef funcName,
                                     mlir::StringRef allocName,
                                     mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<Op>(context, benefit), funcName(funcName),
        allocName(allocName) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    LowLFHEToConcreteCAPITypeConverter typeConverter;
    auto errType = mlir::IndexType::get(rewriter.getContext());
    // Insert forward declaration of the operator function
    {
      mlir::SmallVector<mlir::Type, 4> operands{errType,
                                                op->getResultTypes().front()};
      for (auto ty : op->getOperandTypes()) {
        operands.push_back(typeConverter.convertType(ty));
      }
      auto funcType =
          mlir::FunctionType::get(rewriter.getContext(), operands, {});
      if (insertForwardDeclaration(op, rewriter, funcName, funcType).failed()) {
        return mlir::failure();
      }
    }
    // Insert forward declaration of the alloc function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(), {errType, rewriter.getIndexType()},
          {op->getResultTypes().front()});
      if (insertForwardDeclaration(op, rewriter, allocName, funcType)
              .failed()) {
        return mlir::failure();
      }
    }
    mlir::Type resultType = op->getResultTypes().front();
    auto lweResultType =
        resultType.cast<mlir::zamalang::LowLFHE::LweCiphertextType>();
    // Replace the operation with a call to the `funcName`
    {
      // Create the err value
      auto errOp = rewriter.create<mlir::ConstantOp>(op.getLoc(),
                                                     rewriter.getIndexAttr(0));
      // Add the call to the allocation
      auto lweSizeOp = rewriter.create<mlir::ConstantOp>(
          op.getLoc(), rewriter.getIndexAttr(lweResultType.getSize()));
      mlir::SmallVector<mlir::Value> allocOperands{errOp, lweSizeOp};
      auto alloc = rewriter.replaceOpWithNewOp<mlir::CallOp>(
          op, allocName, op.getType(), allocOperands);

      // Add err and allocated value to operands
      mlir::SmallVector<mlir::Value, 4> newOperands{errOp, alloc.getResult(0)};
      for (auto operand : op->getOperands()) {
        newOperands.push_back(operand);
      }
      rewriter.create<mlir::CallOp>(op.getLoc(), funcName, mlir::TypeRange{},
                                    newOperands);
    }
    return mlir::success();
  };

private:
  std::string funcName;
  std::string allocName;
};

struct LowLFHEZeroOpPattern
    : public mlir::OpRewritePattern<mlir::zamalang::LowLFHE::ZeroLWEOp> {
  LowLFHEZeroOpPattern(mlir::MLIRContext *context,
                       mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::zamalang::LowLFHE::ZeroLWEOp>(context,
                                                                   benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::zamalang::LowLFHE::ZeroLWEOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto allocName = "allocate_lwe_ciphertext_u64";
    auto errType = mlir::IndexType::get(rewriter.getContext());
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(), {errType, rewriter.getIndexType()},
          {op->getResultTypes().front()});
      if (insertForwardDeclaration(op, rewriter, allocName, funcType)
              .failed()) {
        return mlir::failure();
      }
    }
    // Replace the operation with a call to the `funcName`
    {
      mlir::Type resultType = op->getResultTypes().front();
      auto lweResultType =
          resultType.cast<mlir::zamalang::LowLFHE::LweCiphertextType>();
      // Create the err value
      auto errOp = rewriter.create<mlir::ConstantOp>(op.getLoc(),
                                                     rewriter.getIndexAttr(0));
      // Add the call to the allocation
      auto lweSizeOp = rewriter.create<mlir::ConstantOp>(
          op.getLoc(), rewriter.getIndexAttr(lweResultType.getSize()));
      mlir::SmallVector<mlir::Value> allocOperands{errOp, lweSizeOp};
      auto alloc = rewriter.replaceOpWithNewOp<mlir::CallOp>(
          op, allocName, op.getType(), allocOperands);
    }
    return mlir::success();
  };
};

struct LowLFHEEncodeIntOpPattern
    : public mlir::OpRewritePattern<mlir::zamalang::LowLFHE::EncodeIntOp> {
  LowLFHEEncodeIntOpPattern(mlir::MLIRContext *context,
                            mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::zamalang::LowLFHE::EncodeIntOp>(context,
                                                                     benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::zamalang::LowLFHE::EncodeIntOp op,
                  mlir::PatternRewriter &rewriter) const override {
    {
      mlir::Value castedInt = rewriter.create<mlir::ZeroExtendIOp>(
          op.getLoc(), rewriter.getIntegerType(64), op->getOperands().front());
      mlir::Value constantShiftOp = rewriter.create<mlir::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(64 - op.getType().getP()));

      mlir::Type resultType = rewriter.getIntegerType(64);
      rewriter.replaceOpWithNewOp<mlir::ShiftLeftOp>(op, resultType, castedInt,
                                                     constantShiftOp);
    }
    return mlir::success();
  };
};

struct LowLFHEIntToCleartextOpPattern
    : public mlir::OpRewritePattern<mlir::zamalang::LowLFHE::IntToCleartextOp> {
  LowLFHEIntToCleartextOpPattern(mlir::MLIRContext *context,
                                 mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::zamalang::LowLFHE::IntToCleartextOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::zamalang::LowLFHE::IntToCleartextOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value castedInt = rewriter.replaceOpWithNewOp<mlir::ZeroExtendIOp>(
        op, rewriter.getIntegerType(64), op->getOperands().front());
    return mlir::success();
  };
};

/// Populate the RewritePatternSet with all patterns that rewrite LowLFHE
/// operators to the corresponding function call to the `Concrete C API`.
void populateLowLFHEToConcreteCAPICall(mlir::RewritePatternSet &patterns) {
  patterns.add<LowLFHEOpToConcreteCAPICallPattern<
      mlir::zamalang::LowLFHE::AddLweCiphertextsOp>>(
      patterns.getContext(), "add_lwe_ciphertexts_u64",
      "allocate_lwe_ciphertext_u64");
  patterns.add<LowLFHEOpToConcreteCAPICallPattern<
      mlir::zamalang::LowLFHE::AddPlaintextLweCiphertextOp>>(
      patterns.getContext(), "add_plaintext_lwe_ciphertext_u64",
      "allocate_lwe_ciphertext_u64");
  patterns.add<LowLFHEOpToConcreteCAPICallPattern<
      mlir::zamalang::LowLFHE::MulCleartextLweCiphertextOp>>(
      patterns.getContext(), "mul_cleartext_lwe_ciphertext_u64",
      "allocate_lwe_ciphertext_u64");
  patterns.add<LowLFHEOpToConcreteCAPICallPattern<
      mlir::zamalang::LowLFHE::NegateLweCiphertextOp>>(
      patterns.getContext(), "negate_lwe_ciphertext_u64",
      "allocate_lwe_ciphertext_u64");
  patterns.add<LowLFHEEncodeIntOpPattern>(patterns.getContext());
  patterns.add<LowLFHEIntToCleartextOpPattern>(patterns.getContext());
  patterns.add<LowLFHEZeroOpPattern>(patterns.getContext());
}

namespace {
struct LowLFHEToConcreteCAPIPass
    : public LowLFHEToConcreteCAPIBase<LowLFHEToConcreteCAPIPass> {
  void runOnOperation() final;
};
} // namespace

void LowLFHEToConcreteCAPIPass::runOnOperation() {
  // Setup the conversion target.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<mlir::zamalang::LowLFHE::LowLFHEDialect>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::StandardOpsDialect,
                         mlir::memref::MemRefDialect>();

  // Setup rewrite patterns
  mlir::RewritePatternSet patterns(&getContext());
  populateLowLFHEToConcreteCAPICall(patterns);

  // Apply the conversion
  mlir::ModuleOp op = getOperation();
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

namespace mlir {
namespace zamalang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLowLFHEToConcreteCAPIPass() {
  return std::make_unique<LowLFHEToConcreteCAPIPass>();
}
} // namespace zamalang
} // namespace mlir