// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Tools.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"

namespace {

namespace BConcrete = mlir::concretelang::BConcrete;
namespace arith = mlir::arith;
namespace func = mlir::func;
namespace memref = mlir::memref;

char memref_add_lwe_ciphertexts_u64[] = "memref_add_lwe_ciphertexts_u64";
char memref_add_plaintext_lwe_ciphertext_u64[] =
    "memref_add_plaintext_lwe_ciphertext_u64";
char memref_mul_cleartext_lwe_ciphertext_u64[] =
    "memref_mul_cleartext_lwe_ciphertext_u64";
char memref_negate_lwe_ciphertext_u64[] = "memref_negate_lwe_ciphertext_u64";
char memref_keyswitch_lwe_u64[] = "memref_keyswitch_lwe_u64";
char memref_bootstrap_lwe_u64[] = "memref_bootstrap_lwe_u64";
char memref_batched_keyswitch_lwe_u64[] = "memref_batched_keyswitch_lwe_u64";
char memref_batched_bootstrap_lwe_u64[] = "memref_batched_bootstrap_lwe_u64";

char memref_keyswitch_async_lwe_u64[] = "memref_keyswitch_async_lwe_u64";
char memref_bootstrap_async_lwe_u64[] = "memref_bootstrap_async_lwe_u64";
char memref_await_future[] = "memref_await_future";
char memref_keyswitch_lwe_cuda_u64[] = "memref_keyswitch_lwe_cuda_u64";
char memref_bootstrap_lwe_cuda_u64[] = "memref_bootstrap_lwe_cuda_u64";
char memref_batched_keyswitch_lwe_cuda_u64[] =
    "memref_batched_keyswitch_lwe_cuda_u64";
char memref_batched_bootstrap_lwe_cuda_u64[] =
    "memref_batched_bootstrap_lwe_cuda_u64";
char memref_expand_lut_in_trivial_glwe_ct_u64[] =
    "memref_expand_lut_in_trivial_glwe_ct_u64";

char memref_wop_pbs_crt_buffer[] = "memref_wop_pbs_crt_buffer";

mlir::Type getDynamicMemrefWithUnknownOffset(mlir::RewriterBase &rewriter,
                                             size_t rank) {
  std::vector<int64_t> shape(rank, -1);
  mlir::AffineExpr expr = rewriter.getAffineSymbolExpr(0);
  for (size_t i = 0; i < rank; i++) {
    expr = expr +
           (rewriter.getAffineDimExpr(i) * rewriter.getAffineSymbolExpr(i + 1));
  }
  return mlir::MemRefType::get(
      shape, rewriter.getI64Type(),
      mlir::AffineMap::get(rank, rank + 1, expr, rewriter.getContext()));
}

// Returns `memref.cast %0 : memref<...xAxT> to memref<...x?xT>`
mlir::Value getCastedMemRef(mlir::RewriterBase &rewriter, mlir::Value value) {
  mlir::Type valueType = value.getType();

  if (auto memrefTy = valueType.dyn_cast_or_null<mlir::MemRefType>()) {
    return rewriter.create<mlir::memref::CastOp>(
        value.getLoc(),
        getDynamicMemrefWithUnknownOffset(rewriter, memrefTy.getShape().size()),
        value);
  } else {
    return value;
  }
}

mlir::LogicalResult insertForwardDeclarationOfTheCAPI(
    mlir::Operation *op, mlir::RewriterBase &rewriter, char const *funcName) {

  auto memref1DType = getDynamicMemrefWithUnknownOffset(rewriter, 1);
  auto memref2DType = getDynamicMemrefWithUnknownOffset(rewriter, 2);
  auto futureType =
      mlir::concretelang::RT::FutureType::get(rewriter.getIndexType());
  auto contextType =
      mlir::concretelang::Concrete::ContextType::get(rewriter.getContext());
  auto i32Type = rewriter.getI32Type();

  mlir::FunctionType funcType;

  if (funcName == memref_add_lwe_ciphertexts_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(), {memref1DType, memref1DType, memref1DType}, {});
  } else if (funcName == memref_add_plaintext_lwe_ciphertext_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {memref1DType, memref1DType, rewriter.getI64Type()}, {});
  } else if (funcName == memref_mul_cleartext_lwe_ciphertext_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {memref1DType, memref1DType, rewriter.getI64Type()}, {});
  } else if (funcName == memref_negate_lwe_ciphertext_u64) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {memref1DType, memref1DType}, {});
  } else if (funcName == memref_keyswitch_lwe_u64 ||
             funcName == memref_keyswitch_lwe_cuda_u64) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {memref1DType, memref1DType, i32Type,
                                        i32Type, i32Type, i32Type, contextType},
                                       {});
  } else if (funcName == memref_bootstrap_lwe_u64 ||
             funcName == memref_bootstrap_lwe_cuda_u64) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {memref1DType, memref1DType,
                                        memref1DType, i32Type, i32Type, i32Type,
                                        i32Type, i32Type, i32Type, contextType},
                                       {});
  } else if (funcName == memref_keyswitch_async_lwe_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(), {memref1DType, memref1DType, contextType},
        {futureType});
  } else if (funcName == memref_bootstrap_async_lwe_u64) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {memref1DType, memref1DType,
                                        memref1DType, i32Type, i32Type, i32Type,
                                        i32Type, i32Type, i32Type, contextType},
                                       {futureType});
  } else if (funcName == memref_batched_keyswitch_lwe_u64 ||
             funcName == memref_batched_keyswitch_lwe_cuda_u64) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {memref2DType, memref2DType, i32Type,
                                        i32Type, i32Type, i32Type, contextType},
                                       {});
  } else if (funcName == memref_batched_bootstrap_lwe_u64 ||
             funcName == memref_batched_bootstrap_lwe_cuda_u64) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {memref2DType, memref2DType,
                                        memref1DType, i32Type, i32Type, i32Type,
                                        i32Type, i32Type, i32Type, contextType},
                                       {});
  } else if (funcName == memref_await_future) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {memref1DType, futureType, memref1DType, memref1DType}, {});
  } else if (funcName == memref_expand_lut_in_trivial_glwe_ct_u64) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {
                                           memref1DType,
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           memref1DType,
                                       },
                                       {});
  } else if (funcName == memref_wop_pbs_crt_buffer) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {
                                           memref2DType,
                                           memref2DType,
                                           memref1DType,
                                           memref1DType,
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           contextType,
                                       },
                                       {});
  } else {
    op->emitError("unknwon external function") << funcName;
    return mlir::failure();
  }

  return insertForwardDeclaration(op, rewriter, funcName, funcType);
}

template <typename BConcreteOp>
void addNoOperands(BConcreteOp op, mlir::SmallVector<mlir::Value> &operands,
                   mlir::RewriterBase &rewriter) {}

template <typename BConcreteOp, char const *callee>
struct BConcreteToCAPICallPattern : public mlir::OpRewritePattern<BConcreteOp> {
  BConcreteToCAPICallPattern(
      ::mlir::MLIRContext *context,
      std::function<void(BConcreteOp bOp, llvm::SmallVector<mlir::Value> &,
                         mlir::RewriterBase &)>
          addOperands = addNoOperands<BConcreteOp>,
      mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<BConcreteOp>(context, benefit),
        addOperands(addOperands) {}

  ::mlir::LogicalResult
  matchAndRewrite(BConcreteOp bOp,
                  ::mlir::PatternRewriter &rewriter) const override {

    // Create the operands
    mlir::SmallVector<mlir::Value> operands;
    // For all tensor operand get the corresponding casted buffer
    for (auto &operand : bOp->getOpOperands()) {
      mlir::Type type = operand.get().getType();
      if (!type.isa<mlir::MemRefType>()) {
        operands.push_back(operand.get());
      } else {
        operands.push_back(getCastedMemRef(rewriter, operand.get()));
      }
    }

    // append additional argument
    addOperands(bOp, operands, rewriter);

    // Insert forward declaration of the function
    if (insertForwardDeclarationOfTheCAPI(bOp, rewriter, callee).failed()) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(bOp, callee, mlir::TypeRange{},
                                              operands);

    return ::mlir::success();
  };

private:
  std::function<void(BConcreteOp bOp, llvm::SmallVector<mlir::Value> &,
                     mlir::RewriterBase &)>
      addOperands;
};

template <typename KeySwitchOp>
void keyswitchAddOperands(KeySwitchOp op,
                          mlir::SmallVector<mlir::Value> &operands,
                          mlir::RewriterBase &rewriter) {
  // level
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.levelAttr()));
  // base_log
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.baseLogAttr()));
  // lwe_dim_in
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.lwe_dim_inAttr()));
  // lwe_dim_out
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.lwe_dim_outAttr()));
  // context
  operands.push_back(getContextArgument(op));
}

template <typename BootstrapOp>
void bootstrapAddOperands(BootstrapOp op,
                          mlir::SmallVector<mlir::Value> &operands,
                          mlir::RewriterBase &rewriter) {
  // input_lwe_dim
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.inputLweDimAttr()));
  // poly_size
  operands.push_back(
      rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), op.polySizeAttr()));
  // level
  operands.push_back(
      rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), op.levelAttr()));
  // base_log
  operands.push_back(
      rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), op.baseLogAttr()));
  // glwe_dim
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.glweDimensionAttr()));
  // out_precision
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.outPrecisionAttr()));
  // context
  operands.push_back(getContextArgument(op));
}

void wopPBSAddOperands(BConcrete::WopPBSCRTLweBufferOp op,
                       mlir::SmallVector<mlir::Value> &operands,
                       mlir::RewriterBase &rewriter) {
  mlir::Type crtType = mlir::RankedTensorType::get(
      {(int)op.crtDecompositionAttr().size()}, rewriter.getI64Type());
  std::vector<int64_t> values;
  for (auto a : op.crtDecomposition()) {
    values.push_back(a.cast<mlir::IntegerAttr>().getValue().getZExtValue());
  }
  auto attr = rewriter.getI64TensorAttr(values);
  auto x = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), attr, crtType);
  auto globalMemref = mlir::bufferization::getGlobalFor(x, 0);
  rewriter.eraseOp(x);
  assert(!failed(globalMemref));

  auto globalRef = rewriter.create<memref::GetGlobalOp>(
      op.getLoc(), (*globalMemref).type(), (*globalMemref).getName());
  operands.push_back(getCastedMemRef(rewriter, globalRef));

  //   lwe_small_size
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.packingKeySwitchInputLweDimensionAttr()));
  // cbs_level_count
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.circuitBootstrapLevelAttr()));
  // cbs_base_log
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.circuitBootstrapBaseLogAttr()));
  // polynomial_size
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.packingKeySwitchoutputPolynomialSizeAttr()));
  // context
  operands.push_back(getContextArgument(op));
}

struct BConcreteToCAPIPass : public BConcreteToCAPIBase<BConcreteToCAPIPass> {

  BConcreteToCAPIPass(bool gpu) : gpu(gpu) {}

  void runOnOperation() override {
    auto op = this->getOperation();

    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    // Mark ops from the target dialect as legal operations
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();

    // Make sure that no ops from `FHE` remain after the lowering
    target.addIllegalDialect<BConcrete::BConcreteDialect>();

    // Add patterns to transform BConcrete operators to CAPI call
    patterns.add<BConcreteToCAPICallPattern<BConcrete::AddLweBufferOp,
                                            memref_add_lwe_ciphertexts_u64>>(
        &getContext());
    patterns.add<
        BConcreteToCAPICallPattern<BConcrete::AddPlaintextLweBufferOp,
                                   memref_add_plaintext_lwe_ciphertext_u64>>(
        &getContext());
    patterns.add<
        BConcreteToCAPICallPattern<BConcrete::MulCleartextLweBufferOp,
                                   memref_mul_cleartext_lwe_ciphertext_u64>>(
        &getContext());
    patterns.add<BConcreteToCAPICallPattern<BConcrete::NegateLweBufferOp,
                                            memref_negate_lwe_ciphertext_u64>>(
        &getContext());
    if (gpu) {
      patterns.add<BConcreteToCAPICallPattern<BConcrete::KeySwitchLweBufferOp,
                                              memref_keyswitch_lwe_cuda_u64>>(
          &getContext(), keyswitchAddOperands<BConcrete::KeySwitchLweBufferOp>);
      patterns.add<BConcreteToCAPICallPattern<BConcrete::BootstrapLweBufferOp,
                                              memref_bootstrap_lwe_cuda_u64>>(
          &getContext(), bootstrapAddOperands<BConcrete::BootstrapLweBufferOp>);
      patterns.add<
          BConcreteToCAPICallPattern<BConcrete::BatchedKeySwitchLweBufferOp,
                                     memref_batched_keyswitch_lwe_cuda_u64>>(
          &getContext(),
          keyswitchAddOperands<BConcrete::BatchedKeySwitchLweBufferOp>);
      patterns.add<
          BConcreteToCAPICallPattern<BConcrete::BatchedBootstrapLweBufferOp,
                                     memref_batched_bootstrap_lwe_cuda_u64>>(
          &getContext(),
          bootstrapAddOperands<BConcrete::BatchedBootstrapLweBufferOp>);
    } else {
      patterns.add<BConcreteToCAPICallPattern<BConcrete::KeySwitchLweBufferOp,
                                              memref_keyswitch_lwe_u64>>(
          &getContext(), keyswitchAddOperands<BConcrete::KeySwitchLweBufferOp>);
      patterns.add<BConcreteToCAPICallPattern<BConcrete::BootstrapLweBufferOp,
                                              memref_bootstrap_lwe_u64>>(
          &getContext(), bootstrapAddOperands<BConcrete::BootstrapLweBufferOp>);
      patterns.add<
          BConcreteToCAPICallPattern<BConcrete::BatchedKeySwitchLweBufferOp,
                                     memref_batched_keyswitch_lwe_u64>>(
          &getContext(),
          keyswitchAddOperands<BConcrete::BatchedKeySwitchLweBufferOp>);
      patterns.add<
          BConcreteToCAPICallPattern<BConcrete::BatchedBootstrapLweBufferOp,
                                     memref_batched_bootstrap_lwe_u64>>(
          &getContext(),
          bootstrapAddOperands<BConcrete::BatchedBootstrapLweBufferOp>);
    }

    patterns.add<BConcreteToCAPICallPattern<BConcrete::WopPBSCRTLweBufferOp,
                                            memref_wop_pbs_crt_buffer>>(
        &getContext(), wopPBSAddOperands);

    // Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }

private:
  bool gpu;
};

} // namespace

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertBConcreteToCAPIPass(bool gpu) {
  return std::make_unique<BConcreteToCAPIPass>(gpu);
}
} // namespace concretelang
} // namespace mlir
