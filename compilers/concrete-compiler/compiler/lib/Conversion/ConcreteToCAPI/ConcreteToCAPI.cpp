// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Tools.h"
#include "concretelang/Conversion/Utils/Utils.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"

namespace {

namespace Concrete = mlir::concretelang::Concrete;
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
char memref_batched_add_lwe_ciphertexts_u64[] =
    "memref_batched_add_lwe_ciphertexts_u64";
char memref_batched_add_plaintext_lwe_ciphertext_u64[] =
    "memref_batched_add_plaintext_lwe_ciphertext_u64";
char memref_batched_add_plaintext_cst_lwe_ciphertext_u64[] =
    "memref_batched_add_plaintext_cst_lwe_ciphertext_u64";
char memref_batched_mul_cleartext_lwe_ciphertext_u64[] =
    "memref_batched_mul_cleartext_lwe_ciphertext_u64";
char memref_batched_mul_cleartext_cst_lwe_ciphertext_u64[] =
    "memref_batched_mul_cleartext_cst_lwe_ciphertext_u64";
char memref_batched_negate_lwe_ciphertext_u64[] =
    "memref_batched_negate_lwe_ciphertext_u64";
char memref_batched_keyswitch_lwe_u64[] = "memref_batched_keyswitch_lwe_u64";
char memref_batched_bootstrap_lwe_u64[] = "memref_batched_bootstrap_lwe_u64";
char memref_batched_mapped_bootstrap_lwe_u64[] =
    "memref_batched_mapped_bootstrap_lwe_u64";

char memref_keyswitch_async_lwe_u64[] = "memref_keyswitch_async_lwe_u64";
char memref_bootstrap_async_lwe_u64[] = "memref_bootstrap_async_lwe_u64";
char memref_await_future[] = "memref_await_future";
char memref_keyswitch_lwe_cuda_u64[] = "memref_keyswitch_lwe_cuda_u64";
char memref_bootstrap_lwe_cuda_u64[] = "memref_bootstrap_lwe_cuda_u64";
char memref_batched_keyswitch_lwe_cuda_u64[] =
    "memref_batched_keyswitch_lwe_cuda_u64";
char memref_batched_bootstrap_lwe_cuda_u64[] =
    "memref_batched_bootstrap_lwe_cuda_u64";
char memref_batched_mapped_bootstrap_lwe_cuda_u64[] =
    "memref_batched_mapped_bootstrap_lwe_cuda_u64";
char memref_expand_lut_in_trivial_glwe_ct_u64[] =
    "memref_expand_lut_in_trivial_glwe_ct_u64";

char memref_wop_pbs_crt_buffer[] = "memref_wop_pbs_crt_buffer";

char memref_encode_plaintext_with_crt[] = "memref_encode_plaintext_with_crt";
char memref_encode_expand_lut_for_bootstrap[] =
    "memref_encode_expand_lut_for_bootstrap";
char memref_encode_lut_for_crt_woppbs[] = "memref_encode_lut_for_crt_woppbs";
char memref_trace[] = "memref_trace";

mlir::LogicalResult insertForwardDeclarationOfTheCAPI(
    mlir::Operation *op, mlir::RewriterBase &rewriter, char const *funcName) {

  auto memref1DType =
      mlir::concretelang::getDynamicMemrefWithUnknownOffset(rewriter, 1);
  auto memref2DType =
      mlir::concretelang::getDynamicMemrefWithUnknownOffset(rewriter, 2);
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
    funcType =
        mlir::FunctionType::get(rewriter.getContext(),
                                {memref1DType, memref1DType, i32Type, i32Type,
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
    // Todo Answer this question: Isn't it dead ?
    funcType = mlir::FunctionType::get(
        rewriter.getContext(), {memref1DType, memref1DType, contextType},
        {futureType});
  } else if (funcName == memref_bootstrap_async_lwe_u64) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {memref1DType, memref1DType,
                                        memref1DType, i32Type, i32Type, i32Type,
                                        i32Type, i32Type, i32Type, contextType},
                                       {futureType});
  } else if (funcName == memref_batched_add_lwe_ciphertexts_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(), {memref2DType, memref2DType, memref2DType}, {});
  } else if (funcName == memref_batched_add_plaintext_lwe_ciphertext_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(), {memref2DType, memref2DType, memref1DType}, {});
  } else if (funcName == memref_batched_add_plaintext_cst_lwe_ciphertext_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {memref2DType, memref2DType, rewriter.getI64Type()}, {});
  } else if (funcName == memref_batched_mul_cleartext_lwe_ciphertext_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(), {memref2DType, memref2DType, memref1DType}, {});
  } else if (funcName == memref_batched_mul_cleartext_cst_lwe_ciphertext_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {memref2DType, memref2DType, rewriter.getI64Type()}, {});
  } else if (funcName == memref_batched_negate_lwe_ciphertext_u64) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {memref2DType, memref2DType}, {});
  } else if (funcName == memref_batched_keyswitch_lwe_u64 ||
             funcName == memref_batched_keyswitch_lwe_cuda_u64) {
    funcType =
        mlir::FunctionType::get(rewriter.getContext(),
                                {memref2DType, memref2DType, i32Type, i32Type,
                                 i32Type, i32Type, i32Type, contextType},
                                {});
  } else if (funcName == memref_batched_bootstrap_lwe_u64 ||
             funcName == memref_batched_bootstrap_lwe_cuda_u64) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {memref2DType, memref2DType,
                                        memref1DType, i32Type, i32Type, i32Type,
                                        i32Type, i32Type, i32Type, contextType},
                                       {});
  } else if (funcName == memref_batched_mapped_bootstrap_lwe_u64 ||
             funcName == memref_batched_mapped_bootstrap_lwe_cuda_u64) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {memref2DType, memref2DType,
                                        memref2DType, i32Type, i32Type, i32Type,
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
                                           memref2DType,
                                           memref1DType,
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32Type(),
                                           contextType,
                                       },
                                       {});

  } else if (funcName == memref_encode_plaintext_with_crt) {
    funcType = mlir::FunctionType::get(rewriter.getContext(),
                                       {memref1DType, rewriter.getI64Type(),
                                        memref1DType, rewriter.getI64Type()},
                                       {});
  } else if (funcName == memref_encode_expand_lut_for_bootstrap) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {memref1DType, memref1DType, rewriter.getI32Type(),
         rewriter.getI32Type(), rewriter.getI1Type()},
        {});
  } else if (funcName == memref_encode_lut_for_crt_woppbs) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {memref2DType, memref1DType, memref1DType, memref1DType,
         rewriter.getI32Type(), rewriter.getI1Type()},
        {});
  } else if (funcName == memref_trace) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {memref1DType, mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()),
         rewriter.getI32Type(), rewriter.getI32Type()},
        {});
  } else {
    op->emitError("unknown external function") << funcName;
    return mlir::failure();
  }

  return insertForwardDeclaration(op, rewriter, funcName, funcType);
}

template <typename ConcreteOp>
void addNoOperands(ConcreteOp op, mlir::SmallVector<mlir::Value> &operands,
                   mlir::RewriterBase &rewriter) {}

template <typename ConcreteOp, char const *callee>
struct ConcreteToCAPICallPattern : public mlir::OpRewritePattern<ConcreteOp> {
  ConcreteToCAPICallPattern(
      ::mlir::MLIRContext *context,
      std::function<void(ConcreteOp bOp, llvm::SmallVector<mlir::Value> &,
                         mlir::RewriterBase &)>
          addOperands = addNoOperands<ConcreteOp>,
      mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<ConcreteOp>(context, benefit),
        addOperands(addOperands) {}

  ::mlir::LogicalResult
  matchAndRewrite(ConcreteOp bOp,
                  ::mlir::PatternRewriter &rewriter) const override {

    // Create the operands
    mlir::SmallVector<mlir::Value> operands;
    // For all tensor operand get the corresponding casted buffer
    for (auto &operand : bOp->getOpOperands()) {
      mlir::Type type = operand.get().getType();
      if (!type.isa<mlir::MemRefType>()) {
        operands.push_back(operand.get());
      } else {
        operands.push_back(
            mlir::concretelang::getCastedMemRef(rewriter, operand.get()));
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
  std::function<void(ConcreteOp bOp, llvm::SmallVector<mlir::Value> &,
                     mlir::RewriterBase &)>
      addOperands;
};

template <typename KeySwitchOp>
void keyswitchAddOperands(KeySwitchOp op,
                          mlir::SmallVector<mlir::Value> &operands,
                          mlir::RewriterBase &rewriter) {
  // level
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.getLevelAttr()));
  // base_log
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.getBaseLogAttr()));
  // lwe_dim_in
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.getLweDimInAttr()));
  // lwe_dim_out
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.getLweDimOutAttr()));
  // ksk_index
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.getKskIndexAttr()));
  // context
  operands.push_back(getContextArgument(op));
}

template <typename BootstrapOp>
void bootstrapAddOperands(BootstrapOp op,
                          mlir::SmallVector<mlir::Value> &operands,
                          mlir::RewriterBase &rewriter) {
  // input_lwe_dim
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getInputLweDimAttr()));
  // poly_size
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getPolySizeAttr()));
  // level
  operands.push_back(
      rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), op.getLevelAttr()));
  // base_log
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getBaseLogAttr()));
  // glwe_dim
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getGlweDimensionAttr()));
  // bsk_index
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.getBskIndexAttr()));
  // context
  operands.push_back(getContextArgument(op));
}

void wopPBSAddOperands(Concrete::WopPBSCRTLweBufferOp op,
                       mlir::SmallVector<mlir::Value> &operands,
                       mlir::RewriterBase &rewriter) {
  mlir::Type crtType = mlir::RankedTensorType::get(
      {(int)op.getCrtDecompositionAttr().size()}, rewriter.getI64Type());
  std::vector<int64_t> values;
  for (auto a : op.getCrtDecomposition()) {
    values.push_back(a.cast<mlir::IntegerAttr>().getValue().getZExtValue());
  }
  auto attr = rewriter.getI64TensorAttr(values);
  auto x = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), attr, crtType);
  auto globalMemref = mlir::bufferization::getGlobalFor(x, 0);
  rewriter.eraseOp(x);
  assert(!failed(globalMemref));

  auto globalRef = rewriter.create<memref::GetGlobalOp>(
      op.getLoc(), (*globalMemref).getType(), (*globalMemref).getName());
  operands.push_back(mlir::concretelang::getCastedMemRef(rewriter, globalRef));

  //   lwe_small_size
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getPackingKeySwitchInputLweDimensionAttr()));
  // cbs_level_count
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getCircuitBootstrapLevelAttr()));
  // cbs_base_log
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getCircuitBootstrapBaseLogAttr()));

  // ksk_level_count
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getKeyswitchLevelAttr()));
  // ksk_base_log
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getKeyswitchBaseLogAttr()));

  // bsk_level_count
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getBootstrapLevelAttr()));
  // bsk_base_log
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getBootstrapBaseLogAttr()));

  // fpksk_level_count
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getPackingKeySwitchLevelAttr()));
  // fpksk_base_log
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getPackingKeySwitchBaseLogAttr()));

  // polynomial_size
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getPackingKeySwitchoutputPolynomialSizeAttr()));

  // ksk_index
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.getKskIndexAttr()));
  // bsk_index
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.getBskIndexAttr()));
  // pksk_index
  operands.push_back(
      rewriter.create<arith::ConstantOp>(op.getLoc(), op.getPkskIndexAttr()));
  // context
  operands.push_back(getContextArgument(op));
}

void encodePlaintextWithCrtAddOperands(
    Concrete::EncodePlaintextWithCrtBufferOp op,
    mlir::SmallVector<mlir::Value> &operands, mlir::RewriterBase &rewriter) {
  // mods
  mlir::Type modsType = mlir::RankedTensorType::get(
      {(int)op.getModsAttr().size()}, rewriter.getI64Type());
  std::vector<int64_t> modsValues;
  for (auto a : op.getMods()) {
    modsValues.push_back(a.cast<mlir::IntegerAttr>().getValue().getZExtValue());
  }
  auto modsAttr = rewriter.getI64TensorAttr(modsValues);
  auto modsOp =
      rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), modsAttr, modsType);
  auto modsGlobalMemref = mlir::bufferization::getGlobalFor(modsOp, 0);
  rewriter.eraseOp(modsOp);
  assert(!failed(modsGlobalMemref));
  auto modsGlobalRef = rewriter.create<memref::GetGlobalOp>(
      op.getLoc(), (*modsGlobalMemref).getType(),
      (*modsGlobalMemref).getName());
  operands.push_back(
      mlir::concretelang::getCastedMemRef(rewriter, modsGlobalRef));

  // mods_prod
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getModsProdAttr()));
}

void encodeExpandLutForBootstrapAddOperands(
    Concrete::EncodeExpandLutForBootstrapBufferOp op,
    mlir::SmallVector<mlir::Value> &operands, mlir::RewriterBase &rewriter) {
  // poly_size
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getPolySizeAttr()));
  // output bits
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getOutputBitsAttr()));
  // is_signed
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getIsSignedAttr()));
}

void encodeLutForWopPBSAddOperands(Concrete::EncodeLutForCrtWopPBSBufferOp op,
                                   mlir::SmallVector<mlir::Value> &operands,
                                   mlir::RewriterBase &rewriter) {

  // crt_decomposition
  mlir::Type crtDecompositionType = mlir::RankedTensorType::get(
      {(int)op.getCrtDecompositionAttr().size()}, rewriter.getI64Type());
  std::vector<int64_t> crtDecompositionValues;
  for (auto a : op.getCrtDecomposition()) {
    crtDecompositionValues.push_back(
        a.cast<mlir::IntegerAttr>().getValue().getZExtValue());
  }
  auto crtDecompositionAttr = rewriter.getI64TensorAttr(crtDecompositionValues);
  auto crtDecompositionOp = rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), crtDecompositionAttr, crtDecompositionType);
  auto crtDecompositionGlobalMemref =
      mlir::bufferization::getGlobalFor(crtDecompositionOp, 0);
  rewriter.eraseOp(crtDecompositionOp);
  assert(!failed(crtDecompositionGlobalMemref));
  auto crtDecompositionGlobalRef = rewriter.create<memref::GetGlobalOp>(
      op.getLoc(), (*crtDecompositionGlobalMemref).getType(),
      (*crtDecompositionGlobalMemref).getName());
  operands.push_back(
      mlir::concretelang::getCastedMemRef(rewriter, crtDecompositionGlobalRef));

  // crt_bits
  mlir::Type crtBitsType = mlir::RankedTensorType::get(
      {(int)op.getCrtBitsAttr().size()}, rewriter.getI64Type());
  std::vector<int64_t> crtBitsValues;
  for (auto a : op.getCrtBits()) {
    crtBitsValues.push_back(
        a.cast<mlir::IntegerAttr>().getValue().getZExtValue());
  }
  auto crtBitsAttr = rewriter.getI64TensorAttr(crtBitsValues);
  auto crtBitsOp = rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), crtBitsAttr, crtBitsType);
  auto crtBitsGlobalMemref = mlir::bufferization::getGlobalFor(crtBitsOp, 0);
  rewriter.eraseOp(crtBitsOp);
  assert(!failed(crtBitsGlobalMemref));
  auto crtBitsGlobalRef = rewriter.create<memref::GetGlobalOp>(
      op.getLoc(), (*crtBitsGlobalMemref).getType(),
      (*crtBitsGlobalMemref).getName());
  operands.push_back(
      mlir::concretelang::getCastedMemRef(rewriter, crtBitsGlobalRef));
  // modulus_product
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getModulusProductAttr()));
  // is_signed
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op.getIsSignedAttr()));
}

struct ConcreteToCAPIPass : public ConcreteToCAPIBase<ConcreteToCAPIPass> {

  ConcreteToCAPIPass(bool gpu) : gpu(gpu) {}

  void runOnOperation() override {
    auto op = this->getOperation();

    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    // Mark ops from the target dialect as legal operations
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    // Make sure that no ops from `FHE` remain after the lowering
    target.addIllegalDialect<Concrete::ConcreteDialect>();

    // Add patterns to transform Concrete operators to CAPI call
    patterns.add<ConcreteToCAPICallPattern<Concrete::AddLweBufferOp,
                                           memref_add_lwe_ciphertexts_u64>>(
        &getContext());
    patterns.add<
        ConcreteToCAPICallPattern<Concrete::AddPlaintextLweBufferOp,
                                  memref_add_plaintext_lwe_ciphertext_u64>>(
        &getContext());
    patterns.add<
        ConcreteToCAPICallPattern<Concrete::MulCleartextLweBufferOp,
                                  memref_mul_cleartext_lwe_ciphertext_u64>>(
        &getContext());
    patterns.add<ConcreteToCAPICallPattern<Concrete::NegateLweBufferOp,
                                           memref_negate_lwe_ciphertext_u64>>(
        &getContext());
    patterns
        .add<ConcreteToCAPICallPattern<Concrete::EncodePlaintextWithCrtBufferOp,
                                       memref_encode_plaintext_with_crt>>(
            &getContext(), encodePlaintextWithCrtAddOperands);
    patterns.add<
        ConcreteToCAPICallPattern<Concrete::EncodeExpandLutForBootstrapBufferOp,
                                  memref_encode_expand_lut_for_bootstrap>>(
        &getContext(), encodeExpandLutForBootstrapAddOperands);
    patterns
        .add<ConcreteToCAPICallPattern<Concrete::EncodeLutForCrtWopPBSBufferOp,
                                       memref_encode_lut_for_crt_woppbs>>(
            &getContext(), encodeLutForWopPBSAddOperands);
    patterns
        .add<ConcreteToCAPICallPattern<Concrete::BatchedAddLweBufferOp,
                                       memref_batched_add_lwe_ciphertexts_u64>>(
            &getContext());
    patterns.add<ConcreteToCAPICallPattern<
        Concrete::BatchedAddPlaintextLweBufferOp,
        memref_batched_add_plaintext_lwe_ciphertext_u64>>(&getContext());
    patterns.add<ConcreteToCAPICallPattern<
        Concrete::BatchedAddPlaintextCstLweBufferOp,
        memref_batched_add_plaintext_cst_lwe_ciphertext_u64>>(&getContext());
    patterns.add<ConcreteToCAPICallPattern<
        Concrete::BatchedMulCleartextLweBufferOp,
        memref_batched_mul_cleartext_lwe_ciphertext_u64>>(&getContext());
    patterns.add<ConcreteToCAPICallPattern<
        Concrete::BatchedMulCleartextCstLweBufferOp,
        memref_batched_mul_cleartext_cst_lwe_ciphertext_u64>>(&getContext());
    patterns.add<
        ConcreteToCAPICallPattern<Concrete::BatchedNegateLweBufferOp,
                                  memref_batched_negate_lwe_ciphertext_u64>>(
        &getContext());
    if (gpu) {
      patterns.add<ConcreteToCAPICallPattern<Concrete::KeySwitchLweBufferOp,
                                             memref_keyswitch_lwe_cuda_u64>>(
          &getContext(), keyswitchAddOperands<Concrete::KeySwitchLweBufferOp>);
      patterns.add<ConcreteToCAPICallPattern<Concrete::BootstrapLweBufferOp,
                                             memref_bootstrap_lwe_cuda_u64>>(
          &getContext(), bootstrapAddOperands<Concrete::BootstrapLweBufferOp>);
      patterns.add<
          ConcreteToCAPICallPattern<Concrete::BatchedKeySwitchLweBufferOp,
                                    memref_batched_keyswitch_lwe_cuda_u64>>(
          &getContext(),
          keyswitchAddOperands<Concrete::BatchedKeySwitchLweBufferOp>);
      patterns.add<
          ConcreteToCAPICallPattern<Concrete::BatchedBootstrapLweBufferOp,
                                    memref_batched_bootstrap_lwe_cuda_u64>>(
          &getContext(),
          bootstrapAddOperands<Concrete::BatchedBootstrapLweBufferOp>);
      patterns.add<ConcreteToCAPICallPattern<
          Concrete::BatchedMappedBootstrapLweBufferOp,
          memref_batched_mapped_bootstrap_lwe_cuda_u64>>(
          &getContext(),
          bootstrapAddOperands<Concrete::BatchedMappedBootstrapLweBufferOp>);
    } else {
      patterns.add<ConcreteToCAPICallPattern<Concrete::KeySwitchLweBufferOp,
                                             memref_keyswitch_lwe_u64>>(
          &getContext(), keyswitchAddOperands<Concrete::KeySwitchLweBufferOp>);
      patterns.add<ConcreteToCAPICallPattern<Concrete::BootstrapLweBufferOp,
                                             memref_bootstrap_lwe_u64>>(
          &getContext(), bootstrapAddOperands<Concrete::BootstrapLweBufferOp>);
      patterns
          .add<ConcreteToCAPICallPattern<Concrete::BatchedKeySwitchLweBufferOp,
                                         memref_batched_keyswitch_lwe_u64>>(
              &getContext(),
              keyswitchAddOperands<Concrete::BatchedKeySwitchLweBufferOp>);
      patterns
          .add<ConcreteToCAPICallPattern<Concrete::BatchedBootstrapLweBufferOp,
                                         memref_batched_bootstrap_lwe_u64>>(
              &getContext(),
              bootstrapAddOperands<Concrete::BatchedBootstrapLweBufferOp>);
      patterns.add<
          ConcreteToCAPICallPattern<Concrete::BatchedMappedBootstrapLweBufferOp,
                                    memref_batched_mapped_bootstrap_lwe_u64>>(
          &getContext(),
          bootstrapAddOperands<Concrete::BatchedMappedBootstrapLweBufferOp>);
    }

    patterns.add<ConcreteToCAPICallPattern<Concrete::WopPBSCRTLweBufferOp,
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
createConvertConcreteToCAPIPass(bool gpu) {
  return std::make_unique<ConcreteToCAPIPass>(gpu);
}
} // namespace concretelang
} // namespace mlir
