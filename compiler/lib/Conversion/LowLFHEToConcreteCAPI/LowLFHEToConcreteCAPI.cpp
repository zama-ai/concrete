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
                                             mlir::RewriterBase &rewriter,
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

// Set of functions to generate generic types.
// Generic types are used to add forward declarations without a specific type.
// For example, we may need to add LWE ciphertext of different dimensions, or
// allocate them. All the calls to the C API should be done using this generic
// types, and casting should then be performed back to the appropriate type.

inline mlir::zamalang::LowLFHE::LweCiphertextType
getGenericLweCiphertextType(mlir::MLIRContext *context) {
  return mlir::zamalang::LowLFHE::LweCiphertextType::get(context, -1, -1);
}

inline mlir::zamalang::LowLFHE::GlweCiphertextType
getGenericGlweCiphertextType(mlir::MLIRContext *context) {
  return mlir::zamalang::LowLFHE::GlweCiphertextType::get(context);
}

inline mlir::zamalang::LowLFHE::PlaintextType
getGenericPlaintextType(mlir::MLIRContext *context) {
  return mlir::zamalang::LowLFHE::PlaintextType::get(context, -1);
}

inline mlir::zamalang::LowLFHE::PlaintextListType
getGenericPlaintextListType(mlir::MLIRContext *context) {
  return mlir::zamalang::LowLFHE::PlaintextListType::get(context);
}

inline mlir::zamalang::LowLFHE::ForeignPlaintextListType
getGenericForeignPlaintextListType(mlir::MLIRContext *context) {
  return mlir::zamalang::LowLFHE::ForeignPlaintextListType::get(context);
}

inline mlir::zamalang::LowLFHE::CleartextType
getGenericCleartextType(mlir::MLIRContext *context) {
  return mlir::zamalang::LowLFHE::CleartextType::get(context, -1);
}

inline mlir::zamalang::LowLFHE::LweBootstrapKeyType
getGenericLweBootstrapKeyType(mlir::MLIRContext *context) {
  return mlir::zamalang::LowLFHE::LweBootstrapKeyType::get(context);
}

inline mlir::zamalang::LowLFHE::LweKeySwitchKeyType
getGenericLweKeySwitchKeyType(mlir::MLIRContext *context) {
  return mlir::zamalang::LowLFHE::LweKeySwitchKeyType::get(context);
}

// Get the generic version of the type.
// Useful when iterating over a set of types.
mlir::Type getGenericType(mlir::Type baseType) {
  if (baseType.isa<mlir::zamalang::LowLFHE::LweCiphertextType>()) {
    return getGenericLweCiphertextType(baseType.getContext());
  }
  if (baseType.isa<mlir::zamalang::LowLFHE::PlaintextType>()) {
    return getGenericPlaintextType(baseType.getContext());
  }
  if (baseType.isa<mlir::zamalang::LowLFHE::CleartextType>()) {
    return getGenericCleartextType(baseType.getContext());
  }
  return baseType;
}

// Insert all forward declarations needed for the pass.
// Should generalize input and output types for all decalarations, and the
// pattern using them would be resposible for casting them to the appropriate
// type.
mlir::LogicalResult insertForwardDeclarations(mlir::Operation *op,
                                              mlir::IRRewriter &rewriter) {
  auto genericLweCiphertextType =
      getGenericLweCiphertextType(rewriter.getContext());
  auto genericGlweCiphertextType =
      getGenericGlweCiphertextType(rewriter.getContext());
  auto genericPlaintextType = getGenericPlaintextType(rewriter.getContext());
  auto genericPlaintextListType =
      getGenericPlaintextListType(rewriter.getContext());
  auto genericForeignPlaintextList =
      getGenericForeignPlaintextListType(rewriter.getContext());
  auto genericCleartextType = getGenericCleartextType(rewriter.getContext());
  auto genericBSKType = getGenericLweBootstrapKeyType(rewriter.getContext());
  auto genericKSKType = getGenericLweKeySwitchKeyType(rewriter.getContext());
  auto errType = mlir::IndexType::get(rewriter.getContext());

  // Insert forward declaration of allocate lwe ciphertext
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {
                                                errType,
                                                rewriter.getIndexType(),
                                            },

                                            {genericLweCiphertextType});
    if (insertForwardDeclaration(op, rewriter, "allocate_lwe_ciphertext_u64",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the add_lwe_ciphertexts function
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {
                                                errType,
                                                genericLweCiphertextType,
                                                genericLweCiphertextType,
                                                genericLweCiphertextType,
                                            },
                                            {});
    if (insertForwardDeclaration(op, rewriter, "add_lwe_ciphertexts_u64",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the add_plaintext_lwe_ciphertext_u64 function
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {
                                                errType,
                                                genericLweCiphertextType,
                                                genericLweCiphertextType,
                                                genericPlaintextType,
                                            },
                                            {});
    if (insertForwardDeclaration(op, rewriter,
                                 "add_plaintext_lwe_ciphertext_u64", funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the mul_cleartext_lwe_ciphertext_u64 function
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {
                                                errType,
                                                genericLweCiphertextType,
                                                genericLweCiphertextType,
                                                genericCleartextType,
                                            },
                                            {});
    if (insertForwardDeclaration(op, rewriter,
                                 "mul_cleartext_lwe_ciphertext_u64", funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the negate_lwe_ciphertext_u64 function
  {
    auto funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {errType, genericLweCiphertextType, genericLweCiphertextType}, {});
    if (insertForwardDeclaration(op, rewriter, "negate_lwe_ciphertext_u64",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the getBsk function
  {
    auto funcType =
        mlir::FunctionType::get(rewriter.getContext(), {}, {genericBSKType});
    if (insertForwardDeclaration(op, rewriter, "getGlobalBootstrapKey",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the bootstrap function
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {
                                                errType,
                                                genericBSKType,
                                                genericLweCiphertextType,
                                                genericLweCiphertextType,
                                                genericGlweCiphertextType,
                                            },
                                            {});
    if (insertForwardDeclaration(op, rewriter, "bootstrap_lwe_u64", funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the getKsk function
  {
    auto funcType =
        mlir::FunctionType::get(rewriter.getContext(), {}, {genericKSKType});
    if (insertForwardDeclaration(op, rewriter, "getGlobalKeyswitchKey",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the keyswitch function
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {
                                                errType,
                                                // ksk
                                                genericKSKType,
                                                // output ct
                                                genericLweCiphertextType,
                                                // input ct
                                                genericLweCiphertextType,
                                            },
                                            {});
    if (insertForwardDeclaration(op, rewriter, "keyswitch_lwe_u64", funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the alloc_glwe function
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {
                                                errType,
                                                rewriter.getI32Type(),
                                                rewriter.getI32Type(),
                                            },
                                            {genericGlweCiphertextType});
    if (insertForwardDeclaration(op, rewriter, "allocate_glwe_ciphertext_u64",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the alloc_plaintext_list function
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {errType, rewriter.getI32Type()},
                                            {genericPlaintextListType});
    if (insertForwardDeclaration(op, rewriter, "allocate_plaintext_list_u64",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the fill_plaintext_list function
  {
    auto funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {errType, genericPlaintextListType, genericForeignPlaintextList}, {});
    if (insertForwardDeclaration(
            op, rewriter, "fill_plaintext_list_with_expansion_u64", funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the add_plaintext_list_glwe function
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {errType, genericGlweCiphertextType,
                                             genericGlweCiphertextType,
                                             genericPlaintextListType},
                                            {});
    if (insertForwardDeclaration(
            op, rewriter, "add_plaintext_list_glwe_ciphertext_u64", funcType)
            .failed()) {
      return mlir::failure();
    }
  }
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
/// err = arith.constant 0 : i64
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

    mlir::Type resultType = op->getResultTypes().front();
    auto lweResultType =
        resultType.cast<mlir::zamalang::LowLFHE::LweCiphertextType>();
    // Replace the operation with a call to the `funcName`
    {
      // Create the err value
      auto errOp = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getIndexAttr(0));
      // Add the call to the allocation
      auto lweSizeOp = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getIndexAttr(lweResultType.getSize()));
      mlir::SmallVector<mlir::Value> allocOperands{errOp, lweSizeOp};
      auto allocGeneric = rewriter.create<mlir::CallOp>(
          op.getLoc(), allocName,
          getGenericLweCiphertextType(rewriter.getContext()), allocOperands);
      // Construct operands for the operation.
      // errOp doesn't need to be casted to something generic, allocGeneric
      // already is. All the rest will be converted if needed
      mlir::SmallVector<mlir::Value, 4> newOperands{errOp,
                                                    allocGeneric.getResult(0)};
      for (mlir::Value operand : op->getOperands()) {
        mlir::Type operandType = operand.getType();
        mlir::Type castedType = getGenericType(operandType);
        if (castedType == operandType) {
          // Type didn't change, no need for cast
          newOperands.push_back(operand);
        } else {
          // Type changed, need to cast to the generic one
          auto castedOperand = rewriter
                                   .create<mlir::UnrealizedConversionCastOp>(
                                       op.getLoc(), castedType, operand)
                                   .getResult(0);
          newOperands.push_back(castedOperand);
        }
      }
      // The operations called here are known to be inplace, and no need for a
      // return type.
      rewriter.create<mlir::CallOp>(op.getLoc(), funcName, mlir::TypeRange{},
                                    newOperands);
      // cast result value to the appropriate type
      rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
          op, op.getType(), allocGeneric.getResult(0));
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

    mlir::Type resultType = op->getResultTypes().front();
    auto lweResultType =
        resultType.cast<mlir::zamalang::LowLFHE::LweCiphertextType>();
    // Create the err value
    auto errOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(0));
    // Allocate a fresh new ciphertext
    auto lweSizeOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(lweResultType.getSize()));
    mlir::SmallVector<mlir::Value> allocOperands{errOp, lweSizeOp};
    auto allocGeneric = rewriter.create<mlir::CallOp>(
        op.getLoc(), "allocate_lwe_ciphertext_u64",
        getGenericLweCiphertextType(rewriter.getContext()), allocOperands);
    // Cast the result to the appropriate type
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, op.getType(), allocGeneric.getResult(0));

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
      mlir::Value castedInt = rewriter.create<mlir::arith::ExtUIOp>(
          op.getLoc(), rewriter.getIntegerType(64), op->getOperands().front());
      mlir::Value constantShiftOp = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(64 - op.getType().getP()));

      mlir::Type resultType = rewriter.getIntegerType(64);
      rewriter.replaceOpWithNewOp<mlir::arith::ShLIOp>(
          op, resultType, castedInt, constantShiftOp);
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
    rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(
        op, rewriter.getIntegerType(64), op->getOperands().front());
    return mlir::success();
  };
};

// Rewrite the GlweFromTable operation to a series of ops:
// - allocation of two GLWE, one for the addition, and one for storing the
// result
// - allocation of plaintext_list to build the GLWE accumulator
// - build the foreign_plaintext_list using the input table
// - fill the plaintext_list with the foreign_plaintext_list
// - construct the GLWE accumulator by adding the plaintext_list to a freshly
// allocated GLWE
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
    auto errType = mlir::IndexType::get(rewriter.getContext());

    // TODO: move this to insertForwardDeclarations
    // issue: can't define function with tensor<*xtype> that accept ranked
    // tensors

    // Insert forward declaration of the foregin_pt_list function
    {
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(),
          {errType, op->getOperandTypes().front(), rewriter.getI64Type(),
           rewriter.getI32Type()},
          {getGenericForeignPlaintextListType(rewriter.getContext())});
      if (insertForwardDeclaration(
              op, rewriter, "runtime_foreign_plaintext_list_u64", funcType)
              .failed()) {
        return mlir::failure();
      }
    }

    auto errOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(0));
    // allocate two glwe to build accumulator
    auto glweSizeOp =
        rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), op->getAttr("k"));
    auto polySizeOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), op->getAttr("polynomialSize"));
    mlir::SmallVector<mlir::Value> allocGlweOperands{errOp, glweSizeOp,
                                                     polySizeOp};
    // first accumulator would replace the op since it's the returned value
    auto accumulatorOp = rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, "allocate_glwe_ciphertext_u64",
        getGenericGlweCiphertextType(rewriter.getContext()), allocGlweOperands);
    // second accumulator is just needed to build the actual accumulator
    auto _accumulatorOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "allocate_glwe_ciphertext_u64",
        getGenericGlweCiphertextType(rewriter.getContext()), allocGlweOperands);
    // allocate plaintext list
    mlir::SmallVector<mlir::Value> allocPlaintextListOperands{errOp,
                                                              polySizeOp};
    auto plaintextListOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "allocate_plaintext_list_u64",
        getGenericPlaintextListType(rewriter.getContext()),
        allocPlaintextListOperands);
    // create foreign plaintext
    auto rankedTensorType =
        op->getOperandTypes().front().cast<mlir::RankedTensorType>();
    assert(rankedTensorType.getRank() == 1 &&
           "table lookup must be of a single dimension");
    auto sizeOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(),
        rewriter.getI64IntegerAttr(rankedTensorType.getDimSize(0)));
    auto precisionOp =
        rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), op->getAttr("p"));
    mlir::SmallVector<mlir::Value> ForeignPlaintextListOperands{
        errOp, op->getOperand(0), sizeOp, precisionOp};
    auto foreignPlaintextListOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "runtime_foreign_plaintext_list_u64",
        getGenericForeignPlaintextListType(rewriter.getContext()),
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

// Rewrite a BootstrapLweOp with a series of ops:
// - allocate the result LWE ciphertext
// - get the global bootstrapping key
// - use the key and the input accumulator (GLWE) to bootstrap the input
// ciphertext
struct LowLFHEBootstrapLweOpPattern
    : public mlir::OpRewritePattern<mlir::zamalang::LowLFHE::BootstrapLweOp> {
  LowLFHEBootstrapLweOpPattern(mlir::MLIRContext *context,
                               mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::zamalang::LowLFHE::BootstrapLweOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::zamalang::LowLFHE::BootstrapLweOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType = op->getResultTypes().front();
    auto bstOutputSize =
        resultType.cast<mlir::zamalang::LowLFHE::LweCiphertextType>().getSize();
    auto errOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(0));
    // allocate the result lwe ciphertext, should be of a generic type, to cast
    // before return
    auto lweSizeOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(bstOutputSize));
    mlir::SmallVector<mlir::Value> allocLweCtOperands{errOp, lweSizeOp};
    auto allocateGenericLweCtOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "allocate_lwe_ciphertext_u64",
        getGenericLweCiphertextType(rewriter.getContext()), allocLweCtOperands);
    // get bsk
    mlir::SmallVector<mlir::Value> getBskOperands{};
    auto getBskOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "getGlobalBootstrapKey",
        getGenericLweBootstrapKeyType(rewriter.getContext()), getBskOperands);
    // bootstrap
    // cast input ciphertext to a generic type
    mlir::Value lweToBootstrap =
        rewriter
            .create<mlir::UnrealizedConversionCastOp>(
                op.getLoc(), getGenericType(op.getOperand(0).getType()),
                op.getOperand(0))
            .getResult(0);
    // cast input accumulator to a generic type
    mlir::Value accumulator =
        rewriter
            .create<mlir::UnrealizedConversionCastOp>(
                op.getLoc(), getGenericType(op.getOperand(1).getType()),
                op.getOperand(1))
            .getResult(0);
    mlir::SmallVector<mlir::Value> bootstrapOperands{
        errOp, getBskOp.getResult(0), allocateGenericLweCtOp.getResult(0),
        lweToBootstrap, accumulator};
    rewriter.create<mlir::CallOp>(op.getLoc(), "bootstrap_lwe_u64",
                                  mlir::TypeRange({}), bootstrapOperands);
    // Cast result to the appropriate type
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, resultType, allocateGenericLweCtOp.getResult(0));

    return mlir::success();
  };
};

// Rewrite a KeySwitchLweOp with a series of ops:
// - allocate the result LWE ciphertext
// - get the global keyswitch key
// - use the key to keyswitch the input ciphertext
struct LowLFHEKeySwitchLweOpPattern
    : public mlir::OpRewritePattern<mlir::zamalang::LowLFHE::KeySwitchLweOp> {
  LowLFHEKeySwitchLweOpPattern(mlir::MLIRContext *context,
                               mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::zamalang::LowLFHE::KeySwitchLweOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::zamalang::LowLFHE::KeySwitchLweOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto errOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(0));
    // allocate the result lwe ciphertext, should be of a generic type, to cast
    // before return
    auto lweSizeOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(),
        rewriter.getIndexAttr(
            op->getAttr("outputLweSize").cast<mlir::IntegerAttr>().getInt()));
    mlir::SmallVector<mlir::Value> allocLweCtOperands{errOp, lweSizeOp};
    auto allocateGenericLweCtOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "allocate_lwe_ciphertext_u64",
        getGenericLweCiphertextType(rewriter.getContext()), allocLweCtOperands);
    // get ksk
    mlir::SmallVector<mlir::Value> getkskOperands{};
    auto getKskOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "getGlobalKeyswitchKey",
        getGenericLweKeySwitchKeyType(rewriter.getContext()), getkskOperands);
    // keyswitch
    // cast input ciphertext to a generic type
    mlir::Value lweToKeyswitch =
        rewriter
            .create<mlir::UnrealizedConversionCastOp>(
                op.getLoc(), getGenericType(op.getOperand().getType()),
                op.getOperand())
            .getResult(0);
    mlir::SmallVector<mlir::Value> keyswitchOperands{
        errOp, getKskOp.getResult(0), allocateGenericLweCtOp.getResult(0),
        lweToKeyswitch};
    rewriter.create<mlir::CallOp>(op.getLoc(), "keyswitch_lwe_u64",
                                  mlir::TypeRange({}), keyswitchOperands);
    // Cast result to the appropriate type
    auto lweOutputType = op->getResultTypes().front();
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, lweOutputType, allocateGenericLweCtOp.getResult(0));
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
                         mlir::memref::MemRefDialect,
                         mlir::arith::ArithmeticDialect>();

  // Setup rewrite patterns
  mlir::RewritePatternSet patterns(&getContext());
  populateLowLFHEToConcreteCAPICall(patterns);

  // Insert forward declarations
  mlir::ModuleOp op = getOperation();
  mlir::IRRewriter rewriter(&getContext());
  if (insertForwardDeclarations(op, rewriter).failed()) {
    this->signalPassFailure();
  }

  // Apply the conversion
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