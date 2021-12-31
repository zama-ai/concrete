// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#include "mlir//IR/BuiltinTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Support/Constants.h"

class ConcreteToConcreteCAPITypeConverter : public mlir::TypeConverter {

public:
  ConcreteToConcreteCAPITypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](mlir::concretelang::Concrete::PlaintextType type) {
      return mlir::IntegerType::get(type.getContext(), 64);
    });
    addConversion([&](mlir::concretelang::Concrete::CleartextType type) {
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

inline mlir::concretelang::Concrete::LweCiphertextType
getGenericLweCiphertextType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::LweCiphertextType::get(context, -1, -1);
}

inline mlir::concretelang::Concrete::GlweCiphertextType
getGenericGlweCiphertextType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::GlweCiphertextType::get(context);
}

inline mlir::concretelang::Concrete::PlaintextType
getGenericPlaintextType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::PlaintextType::get(context, -1);
}

inline mlir::concretelang::Concrete::PlaintextListType
getGenericPlaintextListType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::PlaintextListType::get(context);
}

inline mlir::concretelang::Concrete::ForeignPlaintextListType
getGenericForeignPlaintextListType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::ForeignPlaintextListType::get(context);
}

inline mlir::concretelang::Concrete::CleartextType
getGenericCleartextType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::CleartextType::get(context, -1);
}

inline mlir::concretelang::Concrete::LweBootstrapKeyType
getGenericLweBootstrapKeyType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::LweBootstrapKeyType::get(context);
}

inline mlir::concretelang::Concrete::LweKeySwitchKeyType
getGenericLweKeySwitchKeyType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::LweKeySwitchKeyType::get(context);
}

// Get the generic version of the type.
// Useful when iterating over a set of types.
mlir::Type getGenericType(mlir::Type baseType) {
  if (baseType.isa<mlir::concretelang::Concrete::LweCiphertextType>()) {
    return getGenericLweCiphertextType(baseType.getContext());
  }
  if (baseType.isa<mlir::concretelang::Concrete::PlaintextType>()) {
    return getGenericPlaintextType(baseType.getContext());
  }
  if (baseType.isa<mlir::concretelang::Concrete::CleartextType>()) {
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
  auto contextType =
      mlir::concretelang::Concrete::ContextType::get(rewriter.getContext());

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
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {contextType}, {genericBSKType});
    if (insertForwardDeclaration(op, rewriter, "get_bootstrap_key", funcType)
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
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {contextType}, {genericKSKType});
    if (insertForwardDeclaration(op, rewriter, "get_keyswitch_key", funcType)
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

/// ConcreteOpToConcreteCAPICallPattern<Op> match the `Op` Operation and
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
struct ConcreteOpToConcreteCAPICallPattern : public mlir::OpRewritePattern<Op> {
  ConcreteOpToConcreteCAPICallPattern(
      mlir::MLIRContext *context, mlir::StringRef funcName,
      mlir::StringRef allocName,
      mlir::PatternBenefit benefit =
          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<Op>(context, benefit), funcName(funcName),
        allocName(allocName) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    ConcreteToConcreteCAPITypeConverter typeConverter;

    mlir::Type resultType = op->getResultTypes().front();
    auto lweResultType =
        resultType.cast<mlir::concretelang::Concrete::LweCiphertextType>();
    // Replace the operation with a call to the `funcName`
    {
      // Create the err value
      auto errOp = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getIndexAttr(0));
      // Get the size from the dimension
      int64_t lweDimension = lweResultType.getDimension();
      int64_t lweSize = lweDimension + 1;
      mlir::Value lweSizeOp = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getIndexAttr(lweSize));
      // Add the call to the allocation
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

struct ConcreteZeroOpPattern
    : public mlir::OpRewritePattern<mlir::concretelang::Concrete::ZeroLWEOp> {
  ConcreteZeroOpPattern(mlir::MLIRContext *context,
                        mlir::PatternBenefit benefit =
                            mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<mlir::concretelang::Concrete::ZeroLWEOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::ZeroLWEOp op,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::Type resultType = op->getResultTypes().front();
    auto lweResultType =
        resultType.cast<mlir::concretelang::Concrete::LweCiphertextType>();
    // Create the err value
    auto errOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(0));
    // Get the size from the dimension
    int64_t lweDimension = lweResultType.getDimension();
    int64_t lweSize = lweDimension + 1;
    mlir::Value lweSizeOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(lweSize));
    // Allocate a fresh new ciphertext
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

struct ConcreteEncodeIntOpPattern
    : public mlir::OpRewritePattern<mlir::concretelang::Concrete::EncodeIntOp> {
  ConcreteEncodeIntOpPattern(mlir::MLIRContext *context,
                             mlir::PatternBenefit benefit =
                                 mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<mlir::concretelang::Concrete::EncodeIntOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::EncodeIntOp op,
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

struct ConcreteIntToCleartextOpPattern
    : public mlir::OpRewritePattern<
          mlir::concretelang::Concrete::IntToCleartextOp> {
  ConcreteIntToCleartextOpPattern(
      mlir::MLIRContext *context,
      mlir::PatternBenefit benefit =
          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<mlir::concretelang::Concrete::IntToCleartextOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::IntToCleartextOp op,
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
    : public mlir::OpRewritePattern<
          mlir::concretelang::Concrete::GlweFromTable> {
  GlweFromTableOpPattern(mlir::MLIRContext *context,
                         mlir::PatternBenefit benefit =
                             mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<mlir::concretelang::Concrete::GlweFromTable>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::GlweFromTable op,
                  mlir::PatternRewriter &rewriter) const override {
    ConcreteToConcreteCAPITypeConverter typeConverter;
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
    // Get the size from the dimension
    int64_t glweDimension =
        op->getAttr("glweDimension").cast<mlir::IntegerAttr>().getInt();
    int64_t glweSize = glweDimension + 1;
    mlir::Value glweSizeOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(glweSize));
    // allocate two glwe to build accumulator
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

mlir::Value getContextArgument(mlir::Operation *op) {
  mlir::Block *block = op->getBlock();
  while (block != nullptr) {
    if (llvm::isa<mlir::FuncOp>(block->getParentOp())) {

      mlir::Value context = block->getArguments().back();

      assert(
          context.getType().isa<mlir::concretelang::Concrete::ContextType>() &&
          "the Concrete.context should be the last argument of the enclosing "
          "function of the op");

      return context;
    }
    block = block->getParentOp()->getBlock();
  }
  assert("can't find a function that enclose the op");
  return nullptr;
}

// Rewrite a BootstrapLweOp with a series of ops:
// - allocate the result LWE ciphertext
// - get the global bootstrapping key
// - use the key and the input accumulator (GLWE) to bootstrap the input
// ciphertext
struct ConcreteBootstrapLweOpPattern
    : public mlir::OpRewritePattern<
          mlir::concretelang::Concrete::BootstrapLweOp> {
  ConcreteBootstrapLweOpPattern(mlir::MLIRContext *context,
                                mlir::PatternBenefit benefit =
                                    mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<mlir::concretelang::Concrete::BootstrapLweOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::BootstrapLweOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType = op->getResultTypes().front();
    auto errOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(0));
    // Get the size from the dimension
    int64_t outputLweDimension =
        resultType.cast<mlir::concretelang::Concrete::LweCiphertextType>()
            .getDimension();
    int64_t outputLweSize = outputLweDimension + 1;
    mlir::Value lweSizeOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(outputLweSize));
    // allocate the result lwe ciphertext, should be of a generic type, to cast
    // before return
    mlir::SmallVector<mlir::Value> allocLweCtOperands{errOp, lweSizeOp};
    auto allocateGenericLweCtOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "allocate_lwe_ciphertext_u64",
        getGenericLweCiphertextType(rewriter.getContext()), allocLweCtOperands);
    // get bsk
    auto getBskOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "get_bootstrap_key",
        getGenericLweBootstrapKeyType(rewriter.getContext()),
        mlir::SmallVector<mlir::Value>{getContextArgument(op)});
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
struct ConcreteKeySwitchLweOpPattern
    : public mlir::OpRewritePattern<
          mlir::concretelang::Concrete::KeySwitchLweOp> {
  ConcreteKeySwitchLweOpPattern(mlir::MLIRContext *context,
                                mlir::PatternBenefit benefit =
                                    mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<mlir::concretelang::Concrete::KeySwitchLweOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::KeySwitchLweOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto errOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(0));
    // Get the size from the dimension
    int64_t lweDimension =
        op.getResult()
            .getType()
            .cast<mlir::concretelang::Concrete::LweCiphertextType>()
            .getDimension();
    int64_t lweSize = lweDimension + 1;
    mlir::Value lweSizeOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(lweSize));
    // allocate the result lwe ciphertext, should be of a generic type, to cast
    // before return
    mlir::SmallVector<mlir::Value> allocLweCtOperands{errOp, lweSizeOp};
    auto allocateGenericLweCtOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "allocate_lwe_ciphertext_u64",
        getGenericLweCiphertextType(rewriter.getContext()), allocLweCtOperands);
    // get ksk
    auto getKskOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), "get_keyswitch_key",
        getGenericLweKeySwitchKeyType(rewriter.getContext()),
        mlir::SmallVector<mlir::Value>{getContextArgument(op)});
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

/// Populate the RewritePatternSet with all patterns that rewrite Concrete
/// operators to the corresponding function call to the `Concrete C API`.
void populateConcreteToConcreteCAPICall(mlir::RewritePatternSet &patterns) {
  patterns.add<ConcreteOpToConcreteCAPICallPattern<
      mlir::concretelang::Concrete::AddLweCiphertextsOp>>(
      patterns.getContext(), "add_lwe_ciphertexts_u64",
      "allocate_lwe_ciphertext_u64");
  patterns.add<ConcreteOpToConcreteCAPICallPattern<
      mlir::concretelang::Concrete::AddPlaintextLweCiphertextOp>>(
      patterns.getContext(), "add_plaintext_lwe_ciphertext_u64",
      "allocate_lwe_ciphertext_u64");
  patterns.add<ConcreteOpToConcreteCAPICallPattern<
      mlir::concretelang::Concrete::MulCleartextLweCiphertextOp>>(
      patterns.getContext(), "mul_cleartext_lwe_ciphertext_u64",
      "allocate_lwe_ciphertext_u64");
  patterns.add<ConcreteOpToConcreteCAPICallPattern<
      mlir::concretelang::Concrete::NegateLweCiphertextOp>>(
      patterns.getContext(), "negate_lwe_ciphertext_u64",
      "allocate_lwe_ciphertext_u64");
  patterns.add<ConcreteEncodeIntOpPattern>(patterns.getContext());
  patterns.add<ConcreteIntToCleartextOpPattern>(patterns.getContext());
  patterns.add<ConcreteZeroOpPattern>(patterns.getContext());
  patterns.add<GlweFromTableOpPattern>(patterns.getContext());
  patterns.add<ConcreteKeySwitchLweOpPattern>(patterns.getContext());
  patterns.add<ConcreteBootstrapLweOpPattern>(patterns.getContext());
}

struct AddRuntimeContextToFuncOpPattern
    : public mlir::OpRewritePattern<mlir::FuncOp> {
  AddRuntimeContextToFuncOpPattern(
      mlir::MLIRContext *context,
      mlir::PatternBenefit benefit =
          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<mlir::FuncOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::FuncOp oldFuncOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::FunctionType oldFuncType = oldFuncOp.getType();

    // Add a Concrete.context to the function signature
    mlir::SmallVector<mlir::Type> newInputs(oldFuncType.getInputs().begin(),
                                            oldFuncType.getInputs().end());
    newInputs.push_back(
        rewriter.getType<mlir::concretelang::Concrete::ContextType>());
    mlir::FunctionType newFuncTy = rewriter.getType<mlir::FunctionType>(
        newInputs, oldFuncType.getResults());
    // Create the new func
    mlir::FuncOp newFuncOp = rewriter.create<mlir::FuncOp>(
        oldFuncOp.getLoc(), oldFuncOp.getName(), newFuncTy);

    // Create the arguments of the new func
    mlir::Region &newFuncBody = newFuncOp.body();
    mlir::Block *newFuncEntryBlock = new mlir::Block();
    newFuncEntryBlock->addArguments(newFuncTy.getInputs());
    newFuncBody.push_back(newFuncEntryBlock);

    // Clone the old body to the new one
    mlir::BlockAndValueMapping map;
    for (auto arg : llvm::enumerate(oldFuncOp.getArguments())) {
      map.map(arg.value(), newFuncEntryBlock->getArgument(arg.index()));
    }
    for (auto &op : oldFuncOp.body().front()) {
      newFuncEntryBlock->push_back(op.clone(map));
    }
    rewriter.eraseOp(oldFuncOp);
    return mlir::success();
  }

  // Legal function are one that are private or has a Concrete.context as last
  // arguments.
  static bool isLegal(mlir::FuncOp funcOp) {
    if (!funcOp.isPublic()) {
      return true;
    }
    // TODO : Don't need to add a runtime context for function that doesn't
    // manipulates concrete types.
    //
    // if (!llvm::any_of(funcOp.getType().getInputs(), [](mlir::Type t) {
    //       if (auto tensorTy = t.dyn_cast_or_null<mlir::TensorType>()) {
    //         t = tensorTy.getElementType();
    //       }
    //       return llvm::isa<mlir::concretelang::Concrete::ConcreteDialect>(
    //           t.getDialect());
    //     })) {
    //   return true;
    // }
    return funcOp.getType().getNumInputs() >= 1 &&
           funcOp.getType()
               .getInputs()
               .back()
               .isa<mlir::concretelang::Concrete::ContextType>();
  }
};

namespace {
struct ConcreteToConcreteCAPIPass
    : public ConcreteToConcreteCAPIBase<ConcreteToConcreteCAPIPass> {
  void runOnOperation() final;
};
} // namespace

void ConcreteToConcreteCAPIPass::runOnOperation() {
  mlir::ModuleOp op = getOperation();

  // First of all add the Concrete.context to the block arguments of function
  // that manipulates ciphertexts.
  {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
      return AddRuntimeContextToFuncOpPattern::isLegal(funcOp);
    });

    patterns.add<AddRuntimeContextToFuncOpPattern>(patterns.getContext());

    // Apply the conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
      return;
    }
  }

  // Insert forward declaration
  mlir::IRRewriter rewriter(&getContext());
  if (insertForwardDeclarations(op, rewriter).failed()) {
    this->signalPassFailure();
  }
  // Rewrite Concrete ops to CallOp to the Concrete C API
  {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    target.addIllegalDialect<mlir::concretelang::Concrete::ConcreteDialect>();
    target.addLegalDialect<mlir::BuiltinDialect, mlir::StandardOpsDialect,
                           mlir::memref::MemRefDialect,
                           mlir::arith::ArithmeticDialect>();

    populateConcreteToConcreteCAPICall(patterns);

    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertConcreteToConcreteCAPIPass() {
  return std::make_unique<ConcreteToConcreteCAPIPass>();
}
} // namespace concretelang
} // namespace mlir
