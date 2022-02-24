// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "mlir//IR/BuiltinTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteDialect.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Support/Constants.h"

namespace {
class BConcreteToBConcreteCAPITypeConverter : public mlir::TypeConverter {

public:
  BConcreteToBConcreteCAPITypeConverter() {
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

inline mlir::Type getGenericLweBufferType(mlir::MLIRContext *context) {
  return mlir::RankedTensorType::get({-1}, mlir::IntegerType::get(context, 64));
}

inline mlir::Type getGenericGlweBufferType(mlir::MLIRContext *context) {
  return mlir::RankedTensorType::get({-1}, mlir::IntegerType::get(context, 64));
}

inline mlir::Type getGenericPlaintextType(mlir::MLIRContext *context) {
  return mlir::IntegerType::get(context, 64);
}

inline mlir::Type getGenericCleartextType(mlir::MLIRContext *context) {
  return mlir::IntegerType::get(context, 64);
}

inline mlir::concretelang::Concrete::PlaintextListType
getGenericPlaintextListType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::PlaintextListType::get(context);
}

inline mlir::concretelang::Concrete::ForeignPlaintextListType
getGenericForeignPlaintextListType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::ForeignPlaintextListType::get(context);
}

inline mlir::concretelang::Concrete::LweKeySwitchKeyType
getGenericLweKeySwitchKeyType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::LweKeySwitchKeyType::get(context);
}

inline mlir::concretelang::Concrete::LweBootstrapKeyType
getGenericLweBootstrapKeyType(mlir::MLIRContext *context) {
  return mlir::concretelang::Concrete::LweBootstrapKeyType::get(context);
}

// Insert all forward declarations needed for the pass.
// Should generalize input and output types for all decalarations, and the
// pattern using them would be resposible for casting them to the appropriate
// type.
mlir::LogicalResult insertForwardDeclarations(mlir::Operation *op,
                                              mlir::IRRewriter &rewriter) {
  auto lweBufferType = getGenericLweBufferType(rewriter.getContext());
  auto plaintextType = getGenericPlaintextType(rewriter.getContext());
  auto cleartextType = getGenericCleartextType(rewriter.getContext());
  auto keySwitchKeyType = getGenericLweKeySwitchKeyType(rewriter.getContext());
  auto bootstrapKeyType = getGenericLweBootstrapKeyType(rewriter.getContext());
  auto contextType =
      mlir::concretelang::Concrete::ContextType::get(rewriter.getContext());

  // Insert forward declaration of the add_lwe_ciphertexts function
  {
    auto funcType = mlir::FunctionType::get(
        rewriter.getContext(), {lweBufferType, lweBufferType, lweBufferType},
        {});
    if (insertForwardDeclaration(op, rewriter, "memref_add_lwe_ciphertexts_u64",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the add_plaintext_lwe_ciphertext function
  {
    auto funcType = mlir::FunctionType::get(
        rewriter.getContext(), {lweBufferType, lweBufferType, plaintextType},
        {});
    if (insertForwardDeclaration(
            op, rewriter, "memref_add_plaintext_lwe_ciphertext_u64", funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the mul_cleartext_lwe_ciphertext function
  {
    auto funcType = mlir::FunctionType::get(
        rewriter.getContext(), {lweBufferType, lweBufferType, cleartextType},
        {});
    if (insertForwardDeclaration(
            op, rewriter, "memref_mul_cleartext_lwe_ciphertext_u64", funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the negate_lwe_ciphertext function
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {lweBufferType, lweBufferType}, {});
    if (insertForwardDeclaration(op, rewriter,
                                 "memref_negate_lwe_ciphertext_u64", funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the memref_keyswitch_lwe_u64 function
  {
    auto funcType = mlir::FunctionType::get(
        rewriter.getContext(), {lweBufferType, lweBufferType, contextType}, {});
    if (insertForwardDeclaration(op, rewriter, "memref_keyswitch_lwe_u64",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the memref_bootstrap_lwe_u64 function
  {
    auto funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {lweBufferType, lweBufferType, lweBufferType, contextType}, {});
    if (insertForwardDeclaration(op, rewriter, "memref_bootstrap_lwe_u64",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }

  // Insert forward declaration of the expand_lut_in_trivial_glwe_ct function
  {
    auto funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {
            getGenericGlweBufferType(rewriter.getContext()),
            rewriter.getI32Type(),
            rewriter.getI32Type(),
            rewriter.getI32Type(),
            mlir::RankedTensorType::get(
                {-1}, mlir::IntegerType::get(rewriter.getContext(), 64)),
        },
        {});
    if (insertForwardDeclaration(
            op, rewriter, "memref_expand_lut_in_trivial_glwe_ct_u64", funcType)
            .failed()) {
      return mlir::failure();
    }
  }

  // Insert forward declaration of the getGlobalKeyswitchKey function
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {contextType}, {keySwitchKeyType});
    if (insertForwardDeclaration(op, rewriter, "get_keyswitch_key_u64",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  // Insert forward declaration of the getGlobalBootstrapKey function
  {
    auto funcType = mlir::FunctionType::get(rewriter.getContext(),
                                            {contextType}, {bootstrapKeyType});
    if (insertForwardDeclaration(op, rewriter, "get_bootstrap_key_u64",
                                 funcType)
            .failed()) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

// For all operands `tensor<Axi64>` replace with
// `%casted = tensor.cast %op : tensor<Axi64> to tensor<?xui64>`
mlir::SmallVector<mlir::Value>
getCastedTensor(mlir::Location loc, mlir::Operation::operand_range operands,
                mlir::PatternRewriter &rewriter) {
  mlir::SmallVector<mlir::Value, 4> newOperands{};
  for (mlir::Value operand : operands) {
    mlir::Type operandType = operand.getType();
    if (operandType.isa<mlir::RankedTensorType>()) {
      mlir::Value castedOp = rewriter.create<mlir::tensor::CastOp>(
          loc, getGenericLweBufferType(rewriter.getContext()), operand);
      newOperands.push_back(castedOp);
    } else {
      newOperands.push_back(operand);
    }
  }
  return std::move(newOperands);
}

// For all operands `tensor<Axi64>` replace with
// `%casted = tensor.cast %op : tensor<Axi64> to tensor<?xui64>`
template <typename Op>
mlir::SmallVector<mlir::Value>
getCastedTensorOperands(Op op, mlir::PatternRewriter &rewriter) {
  return getCastedTensor(op->getLoc(), op->getOperands(), rewriter);
}

/// BConcreteOpToConcreteCAPICallPattern<Op> match the `BConcreteOp`
/// Operation and replace with a call to `funcName`, the funcName should be an
/// external function that was linked later. It insert the forward declaration
/// of the private `funcName` if it not already in the symbol table. The C
/// signature of the function should be `void (out, args..., lweDimension)`, the
/// pattern rewrite:
/// ```
/// "BConcreteOp"(%out, args ...) :
///   (tensor<sizexi64>, tensor<sizexi64>...) -> ()
/// ```
/// to
/// ```
/// %out0 = tensor.cast %out : tensor<sizexi64> to tensor<?xui64>
/// %args = tensor.cast ...
/// call @funcName(%out, args...) : (tensor<?xi64>, tensor<?xi64>...) -> ()
/// ```
template <typename BConcreteOp>
struct ConcreteOpToConcreteCAPICallPattern
    : public mlir::OpRewritePattern<BConcreteOp> {
  ConcreteOpToConcreteCAPICallPattern(mlir::MLIRContext *context,
                                      mlir::StringRef funcName,
                                      mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<BConcreteOp>(context, benefit),
        funcName(funcName) {}

  mlir::LogicalResult
  matchAndRewrite(BConcreteOp op,
                  mlir::PatternRewriter &rewriter) const override {
    BConcreteToBConcreteCAPITypeConverter typeConverter;
    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, funcName, mlir::TypeRange{},
        getCastedTensorOperands<BConcreteOp>(op, rewriter));
    return mlir::success();
  };

private:
  std::string funcName;
};

struct ConcreteEncodeIntOpPattern
    : public mlir::OpRewritePattern<mlir::concretelang::Concrete::EncodeIntOp> {
  ConcreteEncodeIntOpPattern(mlir::MLIRContext *context,
                             mlir::PatternBenefit benefit = 1)
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
  ConcreteIntToCleartextOpPattern(mlir::MLIRContext *context,
                                  mlir::PatternBenefit benefit = 1)
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

// Rewrite pattern that rewrite every
// ```
// "BConcrete.keyswitch_lwe_buffer"(%out, %in) {...}:
//   (tensor<2049xi64>, tensor<2049xi64>) -> ()
// ```
//
// to
//
// ```
// %ksk = call @get_keywswitch_key(%ctx) :
//   (!Concrete.context) -> !Concrete.lwe_key_switch_key
// %out_ = tensor.cast %out : tensor<sizexi64> to tensor<?xi64>
// %in_ = tensor.cast %in : tensor<size'xi64> to tensor<?xi64>
// call @memref_keyswitch_lwe_u64(%ksk, %out_, %in_) :
//   (!Concrete.lwe_key_switch_key, tensor<?xui64>, tensor<?xui64>) -> ()
// ```
struct BConcreteKeySwitchLweOpPattern
    : public mlir::OpRewritePattern<
          mlir::concretelang::BConcrete::KeySwitchLweBufferOp> {
  BConcreteKeySwitchLweOpPattern(mlir::MLIRContext *context,
                                 mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<
            mlir::concretelang::BConcrete::KeySwitchLweBufferOp>(context,
                                                                 benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::BConcrete::KeySwitchLweBufferOp op,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::SmallVector<mlir::Value, 3> operands{};
    operands.append(
        getCastedTensorOperands<
            mlir::concretelang::BConcrete::KeySwitchLweBufferOp>(op, rewriter));
    operands.push_back(getContextArgument(op));

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, "memref_keyswitch_lwe_u64",
                                              mlir::TypeRange({}), operands);
    return mlir::success();
  };
};

// Rewrite pattern that rewrite every
// ```
// "BConcrete.bootstrap_lwe_buffer"(%out, %in, %acc) {...} :
//   (tensor<2049xui64>, tensor<2049xui64>, !Concrete.glwe_ciphertext) -> ()
// ```
//
// to
//
// ```
// %bsk = call @getGlobalBootstrapKey() : () -> !Concrete.lwe_bootstrap_key
// %out_ = tensor.cast %out : tensor<sizexi64> to tensor<?xi64>
// %in_ = tensor.cast %in : tensor<size'xi64> to tensor<?xi64>
// call @memref_bootstrap_lwe_u64(%bsk, %out_, %in_, %acc_) :
//   (!Concrete.lwe_bootstrap_key, tensor<?xi64>, tensor<?xi64>,
//   !Concrete.glwe_ciphertext) -> ()
// ```
struct BConcreteBootstrapLweOpPattern
    : public mlir::OpRewritePattern<
          mlir::concretelang::BConcrete::BootstrapLweBufferOp> {
  BConcreteBootstrapLweOpPattern(mlir::MLIRContext *context,
                                 mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<
            mlir::concretelang::BConcrete::BootstrapLweBufferOp>(context,
                                                                 benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::BConcrete::BootstrapLweBufferOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::SmallVector<mlir::Value, 4> operands{};
    operands.append(
        getCastedTensorOperands<
            mlir::concretelang::BConcrete::BootstrapLweBufferOp>(op, rewriter));
    operands.push_back(getContextArgument(op));
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, "memref_bootstrap_lwe_u64",
                                              mlir::TypeRange({}), operands);
    return mlir::success();
  };
};

// Rewrite pattern that rewrite every
// ```
// "BConcrete.fill_glwe_table"(%glwe, %lut) {glweDimension=1,
// polynomialSize=2048, outPrecision=3} :
//   (tensor<4096xi64>, tensor<32xi64>) -> ()
// ```
//
// to
//
// ```
// %glweDim = arith.constant 1 : i32
// %polySize = arith.constant 2048 : i32
// %outPrecision = arith.constant 3 : i32
// %glwe_ = tensor.cast %glwe : tensor<4096xi64> to tensor<?xi64>
// %lut_ = tensor.cast %lut : tensor<32xi64> to tensor<?xi64>
// call @expand_lut_in_trivial_glwe_ct(%glwe, %polySize, %glweDim,
// %outPrecision, %lut_) :
//   (tensor<?xi64>, i32, i32, tensor<?xi64>) -> ()
// ```
struct BConcreteGlweFromTableOpPattern
    : public mlir::OpRewritePattern<
          mlir::concretelang::BConcrete::FillGlweFromTable> {
  BConcreteGlweFromTableOpPattern(
      mlir::MLIRContext *context,
      mlir::PatternBenefit benefit =
          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<
            mlir::concretelang::BConcrete::FillGlweFromTable>(context,
                                                              benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::BConcrete::FillGlweFromTable op,
                  mlir::PatternRewriter &rewriter) const override {
    BConcreteToBConcreteCAPITypeConverter typeConverter;
    // %glweDim = arith.constant 1 : i32
    // %polySize = arith.constant 2048 : i32
    // %outPrecision = arith.constant 3 : i32

    auto castedOp = getCastedTensorOperands(op, rewriter);

    auto polySizeOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(op.polynomialSize()));
    auto glweDimensionOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(op.glweDimension()));
    auto outPrecisionOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(op.outPrecision()));

    mlir::SmallVector<mlir::Value> newOperands{
        castedOp[0], polySizeOp, glweDimensionOp, outPrecisionOp, castedOp[1]};

    // getCastedTensor(op.getLoc(), newOperands, rewriter);
    // perform operands conversion
    // %glwe_ = tensor.cast %glwe : tensor<4096xi64> to tensor<?xi64>
    // %lut_ = tensor.cast %lut : tensor<32xi64> to tensor<?xi64>

    // call @expand_lut_in_trivial_glwe_ct(%glwe, %polySize, %glweDim,
    // %lut_) :
    //   (tensor<?xi64>, i32, i32, tensor<?xi64>) -> ()

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, "memref_expand_lut_in_trivial_glwe_ct_u64",
        mlir::SmallVector<mlir::Type>{}, newOperands);
    return mlir::success();
  };
};

/// Populate the RewritePatternSet with all patterns that rewrite Concrete
/// operators to the corresponding function call to the `Concrete C API`.
void populateBConcreteToBConcreteCAPICall(mlir::RewritePatternSet &patterns) {
  patterns.add<ConcreteOpToConcreteCAPICallPattern<
      mlir::concretelang::BConcrete::AddLweBuffersOp>>(
      patterns.getContext(), "memref_add_lwe_ciphertexts_u64");
  patterns.add<ConcreteOpToConcreteCAPICallPattern<
      mlir::concretelang::BConcrete::AddPlaintextLweBufferOp>>(
      patterns.getContext(), "memref_add_plaintext_lwe_ciphertext_u64");
  patterns.add<ConcreteOpToConcreteCAPICallPattern<
      mlir::concretelang::BConcrete::MulCleartextLweBufferOp>>(
      patterns.getContext(), "memref_mul_cleartext_lwe_ciphertext_u64");
  patterns.add<ConcreteOpToConcreteCAPICallPattern<
      mlir::concretelang::BConcrete::NegateLweBufferOp>>(
      patterns.getContext(), "memref_negate_lwe_ciphertext_u64");
  patterns.add<ConcreteEncodeIntOpPattern>(patterns.getContext());
  patterns.add<ConcreteIntToCleartextOpPattern>(patterns.getContext());
  patterns.add<BConcreteKeySwitchLweOpPattern>(patterns.getContext());
  patterns.add<BConcreteBootstrapLweOpPattern>(patterns.getContext());
  patterns.add<BConcreteGlweFromTableOpPattern>(patterns.getContext());
}

struct AddRuntimeContextToFuncOpPattern
    : public mlir::OpRewritePattern<mlir::FuncOp> {
  AddRuntimeContextToFuncOpPattern(mlir::MLIRContext *context,
                                   mlir::PatternBenefit benefit = 1)
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
    // manipulates Concrete types.
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
struct BConcreteToBConcreteCAPIPass
    : public BConcreteToBConcreteCAPIBase<BConcreteToBConcreteCAPIPass> {
  void runOnOperation() final;
};
} // namespace

void BConcreteToBConcreteCAPIPass::runOnOperation() {
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

    target.addIllegalDialect<mlir::concretelang::BConcrete::BConcreteDialect>();

    target.addLegalDialect<mlir::BuiltinDialect, mlir::StandardOpsDialect,
                           mlir::tensor::TensorDialect,
                           mlir::arith::ArithmeticDialect>();

    populateBConcreteToBConcreteCAPICall(patterns);

    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }
}
} // namespace

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertBConcreteToBConcreteCAPIPass() {
  return std::make_unique<BConcreteToBConcreteCAPIPass>();
}
} // namespace concretelang
} // namespace mlir