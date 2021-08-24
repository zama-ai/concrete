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

struct GlweFromTableOpPattern
    : public mlir::OpRewritePattern<mlir::zamalang::LowLFHE::GlweFromTable> {
  GlweFromTableOpPattern(mlir::MLIRContext *context,
                         mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::zamalang::LowLFHE::GlweFromTable>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::zamalang::LowLFHE::GlweFromTable op,
                  mlir::PatternRewriter &rewriter) const override {
    LowLFHEToConcreteCAPITypeConverter typeConverter;
    auto errType =
        mlir::MemRefType::get({}, mlir::IndexType::get(rewriter.getContext()));
    // Insert forward declaration of the alloc_glwe function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {
              errType,
              mlir::IntegerType::get(rewriter.getContext(), 32),
              mlir::IntegerType::get(rewriter.getContext(), 32),
          },
          {mlir::zamalang::LowLFHE::GlweCiphertextType::get(
              rewriter.getContext())});
      if (insertForwardDeclaration(op, rewriter, "allocate_glwe_ciphertext_u64",
                                   funcType)
              .failed()) {
        return mlir::failure();
      }
    }
    // Insert forward declaration of the alloc_plaintext_list function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {errType, mlir::IntegerType::get(rewriter.getContext(), 32)},
          {mlir::zamalang::LowLFHE::PlaintextListType::get(
              rewriter.getContext())});
      if (insertForwardDeclaration(op, rewriter, "allocate_plaintext_list_u64",
                                   funcType)
              .failed()) {
        return mlir::failure();
      }
    }

    // Insert forward declaration of the foregin_pt_list function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {errType,
           //  mlir::UnrankedTensorType::get(
           //      mlir::IntegerType::get(rewriter.getContext(), 64)),
           op->getOperandTypes().front(),
           mlir::IntegerType::get(rewriter.getContext(), 64)},
          {mlir::zamalang::LowLFHE::ForeignPlaintextListType::get(
              rewriter.getContext())});
      if (insertForwardDeclaration(op, rewriter, "foreign_plaintext_list_u64",
                                   funcType)
              .failed()) {
        return mlir::failure();
      }
    }

    // Insert forward declaration of the fill_plaintext_list function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {errType,
           mlir::zamalang::LowLFHE::PlaintextListType::get(
               rewriter.getContext()),
           mlir::zamalang::LowLFHE::ForeignPlaintextListType::get(
               rewriter.getContext())},
          {});
      if (insertForwardDeclaration(
              op, rewriter, "fill_plaintext_list_with_expansion_u64", funcType)
              .failed()) {
        return mlir::failure();
      }
    }

    // Insert forward declaration of the add_plaintext_list_glwe function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {errType,
           mlir::zamalang::LowLFHE::GlweCiphertextType::get(
               rewriter.getContext()),
           mlir::zamalang::LowLFHE::GlweCiphertextType::get(
               rewriter.getContext()),
           mlir::zamalang::LowLFHE::PlaintextListType::get(
               rewriter.getContext())},
          {});
      if (insertForwardDeclaration(
              op, rewriter, "add_plaintext_list_glwe_ciphertext_u64", funcType)
              .failed()) {
        return mlir::failure();
      }
    }
    auto errOp = rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), errType);
    // allocate two glwe to build accumulator
    auto glweSizeOp =
        rewriter.create<mlir::ConstantOp>(op.getLoc(), op->getAttr("k"));
    auto polySizeOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), op->getAttr("polynomialSize"));
    mlir::SmallVector<mlir::Value> allocGlweOperands{errOp, glweSizeOp,
                                                     polySizeOp};
    // first accumulator would replace the op since it's the returned value
    auto accumulatorOp = rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, "allocate_glwe_ciphertext_u64",
        mlir::zamalang::LowLFHE::GlweCiphertextType::get(rewriter.getContext()),
        allocGlweOperands);
    // second accumulator is just needed to build the actual accumulator
    auto _accumulatorOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "allocate_glwe_ciphertext_u64",
        mlir::zamalang::LowLFHE::GlweCiphertextType::get(rewriter.getContext()),
        allocGlweOperands);
    // allocate plaintext list
    mlir::SmallVector<mlir::Value> allocPlaintextListOperands{errOp,
                                                              polySizeOp};
    auto plaintextListOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "allocate_plaintext_list_u64",
        mlir::zamalang::LowLFHE::PlaintextListType::get(rewriter.getContext()),
        allocPlaintextListOperands);
    // create foreign plaintext
    auto rankedTensorType =
        op->getOperandTypes().front().cast<mlir::RankedTensorType>();
    if (rankedTensorType.getRank() != 1) {
      llvm::errs() << "table lookup must be of a single dimension";
      return mlir::failure();
    }
    auto sizeOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(
                         mlir::IntegerType::get(rewriter.getContext(), 64),
                         rankedTensorType.getDimSize(0)));
    mlir::SmallVector<mlir::Value> ForeignPlaintextListOperands{
        errOp, op->getOperand(0), sizeOp};
    auto foreignPlaintextListOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "foreign_plaintext_list_u64",
        mlir::zamalang::LowLFHE::ForeignPlaintextListType::get(
            rewriter.getContext()),
        ForeignPlaintextListOperands);
    // fill plaintext list
    mlir::SmallVector<mlir::Value> FillPlaintextListOperands{
        errOp, plaintextListOp.getResult(0),
        foreignPlaintextListOp.getResult(0)};
    rewriter.create<mlir::CallOp>(
        op.getLoc(), "fill_plaintext_list_with_expansion_u64",
        mlir::TypeRange({}), FillPlaintextListOperands);
    // add plaintext list and glwe to build final accumulator for pbs
    mlir::SmallVector<mlir::Value> AddPlaintextListGlweOperands{
        errOp, accumulatorOp.getResult(0), _accumulatorOp.getResult(0),
        plaintextListOp.getResult(0)};
    rewriter.create<mlir::CallOp>(
        op.getLoc(), "add_plaintext_list_glwe_ciphertext_u64",
        mlir::TypeRange({}), AddPlaintextListGlweOperands);
    return mlir::success();
  };
};

// TODO:
// Parameterization
// Get concrete key
struct LowLFHEBootstrapLweOpPattern
    : public mlir::OpRewritePattern<mlir::zamalang::LowLFHE::BootstrapLweOp> {
  LowLFHEBootstrapLweOpPattern(mlir::MLIRContext *context,
                               mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::zamalang::LowLFHE::BootstrapLweOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::zamalang::LowLFHE::BootstrapLweOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto errType =
        mlir::MemRefType::get({}, mlir::IndexType::get(rewriter.getContext()));
    auto lweOperandType = op->getOperandTypes().front();
    // Insert forward declaration of the allocate_bsk_key function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {
              errType,
              // level
              mlir::IntegerType::get(rewriter.getContext(), 32),
              // baselog
              mlir::IntegerType::get(rewriter.getContext(), 32),
              // glwe size
              mlir::IntegerType::get(rewriter.getContext(), 32),
              // lwe size
              mlir::IntegerType::get(rewriter.getContext(), 32),
              // polynomial size
              mlir::IntegerType::get(rewriter.getContext(), 32),
          },
          {mlir::zamalang::LowLFHE::LweBootstrapKeyType::get(
              rewriter.getContext())});
      if (insertForwardDeclaration(op, rewriter,
                                   "allocate_lwe_bootstrap_key_u64", funcType)
              .failed()) {
        return mlir::failure();
      }
    }
    // Insert forward declaration of the allocate_lwe_ct function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {
              errType,
              mlir::IntegerType::get(rewriter.getContext(), 32),
          },
          {lweOperandType});
      if (insertForwardDeclaration(op, rewriter, "allocate_lwe_ciphertext_u64",
                                   funcType)
              .failed()) {
        return mlir::failure();
      }
    }
    // Insert forward declaration of the bootstrap function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {
              errType,
              mlir::zamalang::LowLFHE::LweBootstrapKeyType::get(
                  rewriter.getContext()),
              lweOperandType,
              lweOperandType,
              mlir::zamalang::LowLFHE::GlweCiphertextType::get(
                  rewriter.getContext()),
          },
          {});
      if (insertForwardDeclaration(op, rewriter, "bootstrap_lwe_u64", funcType)
              .failed()) {
        return mlir::failure();
      }
    }

    auto errOp = rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), errType);
    // allocate the result lwe ciphertext
    auto lweSizeOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(),
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(rewriter.getContext(), 32), -1));
    mlir::SmallVector<mlir::Value> allocLweCtOperands{errOp, lweSizeOp};
    auto allocateLweCtOp = rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, "allocate_lwe_ciphertext_u64", lweOperandType, allocLweCtOperands);
    // allocate bsk
    auto decompLevelCountOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(),
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(rewriter.getContext(), 32), -1));
    auto decompBaseLogOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(),
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(rewriter.getContext(), 32), -1));
    auto glweSizeOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(),
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(rewriter.getContext(), 32), -1));
    auto polySizeOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(),
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(rewriter.getContext(), 32), -1));
    mlir::SmallVector<mlir::Value> allocBskOperands{
        errOp,      decompLevelCountOp, decompBaseLogOp,
        glweSizeOp, lweSizeOp,          polySizeOp};
    auto allocateBskOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "allocate_lwe_bootstrap_key_u64",
        mlir::zamalang::LowLFHE::LweBootstrapKeyType::get(
            rewriter.getContext()),
        allocBskOperands);
    // bootstrap
    mlir::SmallVector<mlir::Value> bootstrapOperands{
        errOp, allocateBskOp.getResult(0), allocateLweCtOp.getResult(0),
        op->getOperand(0), op->getOperand(1)};
    rewriter.create<mlir::CallOp>(op.getLoc(), "bootstrap_lwe_u64",
                                  mlir::TypeRange({}), bootstrapOperands);

    return mlir::success();
  };
};

// TODO:
// Parameterization
// Get concrete key
struct LowLFHEKeySwitchLweOpPattern
    : public mlir::OpRewritePattern<mlir::zamalang::LowLFHE::KeySwitchLweOp> {
  LowLFHEKeySwitchLweOpPattern(mlir::MLIRContext *context,
                               mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::zamalang::LowLFHE::KeySwitchLweOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::zamalang::LowLFHE::KeySwitchLweOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto errType =
        mlir::MemRefType::get({}, mlir::IndexType::get(rewriter.getContext()));
    auto lweOperandType = op->getOperandTypes().front();
    // Insert forward declaration of the allocate_bsk_key function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {
              errType,
              // level
              mlir::IntegerType::get(rewriter.getContext(), 32),
              // baselog
              mlir::IntegerType::get(rewriter.getContext(), 32),
              // input lwe size
              mlir::IntegerType::get(rewriter.getContext(), 32),
              // output lwe size
              mlir::IntegerType::get(rewriter.getContext(), 32),
          },
          {mlir::zamalang::LowLFHE::LweKeySwitchKeyType::get(
              rewriter.getContext())});
      if (insertForwardDeclaration(op, rewriter,
                                   "allocate_lwe_keyswitch_key_u64", funcType)
              .failed()) {
        return mlir::failure();
      }
    }
    // Insert forward declaration of the allocate_lwe_ct function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {
              errType,
              mlir::IntegerType::get(rewriter.getContext(), 32),
          },
          {lweOperandType});
      if (insertForwardDeclaration(op, rewriter, "allocate_lwe_ciphertext_u64",
                                   funcType)
              .failed()) {
        return mlir::failure();
      }
    }
    // TODO: build the right type here
    auto lweOutputType = lweOperandType;
    // Insert forward declaration of the keyswitch function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {
              errType,
              // ksk
              mlir::zamalang::LowLFHE::LweKeySwitchKeyType::get(
                  rewriter.getContext()),
              // output ct
              lweOutputType,
              // input ct
              lweOperandType,
          },
          {});
      if (insertForwardDeclaration(op, rewriter, "keyswitch_lwe_u64", funcType)
              .failed()) {
        return mlir::failure();
      }
    }

    auto errOp = rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), errType);
    // allocate the result lwe ciphertext
    auto lweSizeOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(),
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(rewriter.getContext(), 32), -1));
    mlir::SmallVector<mlir::Value> allocLweCtOperands{errOp, lweSizeOp};
    auto allocateLweCtOp = rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, "allocate_lwe_ciphertext_u64", lweOutputType, allocLweCtOperands);
    // allocate ksk
    auto decompLevelCountOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(),
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(rewriter.getContext(), 32), -1));
    auto decompBaseLogOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(),
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(rewriter.getContext(), 32), -1));
    auto inputLweSizeOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(),
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(rewriter.getContext(), 32), -1));
    auto outputLweSizeOp = rewriter.create<mlir::ConstantOp>(
        op.getLoc(),
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(rewriter.getContext(), 32), -1));
    mlir::SmallVector<mlir::Value> allockskOperands{
        errOp, decompLevelCountOp, decompBaseLogOp, inputLweSizeOp,
        outputLweSizeOp};
    auto allocateKskOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "allocate_lwe_keyswitch_key_u64",
        mlir::zamalang::LowLFHE::LweKeySwitchKeyType::get(
            rewriter.getContext()),
        allockskOperands);
    // bootstrap
    mlir::SmallVector<mlir::Value> bootstrapOperands{
        errOp, allocateKskOp.getResult(0), allocateLweCtOp.getResult(0),
        op->getOperand(0)};
    rewriter.create<mlir::CallOp>(op.getLoc(), "keyswitch_lwe_u64",
                                  mlir::TypeRange({}), bootstrapOperands);

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
  patterns.add<GlweFromTableOpPattern>(patterns.getContext());
  patterns.add<LowLFHEKeySwitchLweOpPattern>(patterns.getContext());
  patterns.add<LowLFHEBootstrapLweOpPattern>(patterns.getContext());
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