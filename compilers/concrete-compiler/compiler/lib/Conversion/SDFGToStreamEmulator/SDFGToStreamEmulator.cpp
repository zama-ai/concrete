// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Tools.h"
#include "concretelang/Dialect/SDFG/IR/SDFGDialect.h"
#include "concretelang/Dialect/SDFG/IR/SDFGOps.h"
#include "concretelang/Dialect/SDFG/IR/SDFGTypes.h"
#include "concretelang/Runtime/stream_emulator_api.h"

namespace SDFG = mlir::concretelang::SDFG;

namespace {
struct SDFGToStreamEmulatorPass
    : public SDFGToStreamEmulatorBase<SDFGToStreamEmulatorPass> {
  void runOnOperation() final;
};

char stream_emulator_init[] = "stream_emulator_init";
char stream_emulator_run[] = "stream_emulator_run";
char stream_emulator_delete[] = "stream_emulator_delete";
char stream_emulator_make_memref_add_lwe_ciphertexts_u64_process[] =
    "stream_emulator_make_memref_add_lwe_ciphertexts_u64_process";
char stream_emulator_make_memref_add_plaintext_lwe_ciphertext_u64_process[] =
    "stream_emulator_make_memref_add_plaintext_lwe_ciphertext_u64_process";
char stream_emulator_make_memref_mul_cleartext_lwe_ciphertext_u64_process[] =
    "stream_emulator_make_memref_mul_cleartext_lwe_ciphertext_u64_process";
char stream_emulator_make_memref_negate_lwe_ciphertext_u64_process[] =
    "stream_emulator_make_memref_negate_lwe_ciphertext_u64_process";
char stream_emulator_make_memref_keyswitch_lwe_u64_process[] =
    "stream_emulator_make_memref_keyswitch_lwe_u64_process";
char stream_emulator_make_memref_bootstrap_lwe_u64_process[] =
    "stream_emulator_make_memref_bootstrap_lwe_u64_process";

char stream_emulator_make_memref_batched_add_lwe_ciphertexts_u64_process[] =
    "stream_emulator_make_memref_batched_add_lwe_ciphertexts_u64_process";
char
    stream_emulator_make_memref_batched_add_plaintext_lwe_ciphertext_u64_process
        [] = "stream_emulator_make_memref_batched_add_plaintext_lwe_ciphertext_"
             "u64_process";
char
    stream_emulator_make_memref_batched_add_plaintext_cst_lwe_ciphertext_u64_process
        [] = "stream_emulator_make_memref_batched_add_plaintext_cst_lwe_"
             "ciphertext_u64_process";
char
    stream_emulator_make_memref_batched_mul_cleartext_lwe_ciphertext_u64_process
        [] = "stream_emulator_make_memref_batched_mul_cleartext_lwe_ciphertext_"
             "u64_process";
char
    stream_emulator_make_memref_batched_mul_cleartext_cst_lwe_ciphertext_u64_process
        [] = "stream_emulator_make_memref_batched_mul_cleartext_cst_lwe_"
             "ciphertext_u64_process";
char stream_emulator_make_memref_batched_negate_lwe_ciphertext_u64_process[] =
    "stream_emulator_make_memref_batched_negate_lwe_ciphertext_u64_process";
char stream_emulator_make_memref_batched_keyswitch_lwe_u64_process[] =
    "stream_emulator_make_memref_batched_keyswitch_lwe_u64_process";
char stream_emulator_make_memref_batched_bootstrap_lwe_u64_process[] =
    "stream_emulator_make_memref_batched_bootstrap_lwe_u64_process";
char stream_emulator_make_memref_batched_mapped_bootstrap_lwe_u64_process[] =
    "stream_emulator_make_memref_batched_mapped_bootstrap_lwe_u64_process";

char stream_emulator_make_memref_stream[] =
    "stream_emulator_make_memref_stream";
char stream_emulator_put_memref[] = "stream_emulator_put_memref";
char stream_emulator_make_uint64_stream[] =
    "stream_emulator_make_uint64_stream";
char stream_emulator_put_uint64[] = "stream_emulator_put_uint64";
char stream_emulator_get_uint64[] = "stream_emulator_get_uint64";

char stream_emulator_make_memref_batch_stream[] =
    "stream_emulator_make_memref_batch_stream";
char stream_emulator_put_memref_batch[] = "stream_emulator_put_memref_batch";

mlir::Type getDynamicTensor(mlir::OpBuilder &rewriter, size_t rank) {
  std::vector<int64_t> shape(rank, mlir::ShapedType::kDynamic);
  return mlir::RankedTensorType::get(shape, rewriter.getI64Type());
}

mlir::Type makeDynamicTensorTypes(mlir::OpBuilder &rewriter, mlir::Type oldTy) {
  if (auto ttype = oldTy.dyn_cast_or_null<mlir::TensorType>())
    return getDynamicTensor(rewriter, ttype.getRank());
  if (auto stTy = oldTy.dyn_cast_or_null<SDFG::StreamType>())
    return SDFG::StreamType::get(
        rewriter.getContext(),
        makeDynamicTensorTypes(rewriter, stTy.getElementType()));
  return oldTy;
}

mlir::LogicalResult insertGenericForwardDeclaration(mlir::Operation *op,
                                                    mlir::OpBuilder &rewriter,
                                                    llvm::StringRef funcName,
                                                    mlir::TypeRange opTys,
                                                    mlir::TypeRange resTys) {
  mlir::SmallVector<mlir::Type> operands;
  for (mlir::Type opTy : opTys)
    operands.push_back(makeDynamicTensorTypes(rewriter, opTy));
  mlir::SmallVector<mlir::Type> results;
  for (mlir::Type resTy : resTys)
    results.push_back(makeDynamicTensorTypes(rewriter, resTy));

  mlir::FunctionType funcType =
      mlir::FunctionType::get(rewriter.getContext(), operands, results);
  return insertForwardDeclaration(op, rewriter, funcName, funcType);
}

void castDynamicTensorOps(mlir::Operation *op, mlir::OpBuilder &rewriter,
                          mlir::ValueRange operands,
                          mlir::SmallVector<mlir::Value> &newOps) {
  for (auto val : operands) {
    auto oldTy = val.getType();
    if (auto ttype = oldTy.dyn_cast_or_null<mlir::TensorType>())
      newOps.push_back(rewriter.create<mlir::tensor::CastOp>(
          op->getLoc(), getDynamicTensor(rewriter, ttype.getRank()), val));
    else
      newOps.push_back(val);
  }
}

struct LowerSDFGInit
    : public mlir::OpRewritePattern<mlir::concretelang::SDFG::Init> {
  LowerSDFGInit(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::SDFG::Init>(context,
                                                                 benefit) {}
  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::SDFG::Init initOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    mlir::FunctionType funcType = mlir::FunctionType::get(
        rewriter.getContext(), {}, {SDFG::DFGType::get(rewriter.getContext())});
    if (insertForwardDeclaration(initOp, rewriter, stream_emulator_init,
                                 funcType)
            .failed())
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        initOp, stream_emulator_init,
        mlir::TypeRange{SDFG::DFGType::get(rewriter.getContext())});
    return ::mlir::success();
  };
};

struct LowerSDFGStart
    : public mlir::OpRewritePattern<mlir::concretelang::SDFG::Start> {
  LowerSDFGStart(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::SDFG::Start>(context,
                                                                  benefit) {}
  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::SDFG::Start startOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    mlir::FunctionType funcType = mlir::FunctionType::get(
        rewriter.getContext(), {SDFG::DFGType::get(rewriter.getContext())}, {});
    if (insertForwardDeclaration(startOp, rewriter, stream_emulator_run,
                                 funcType)
            .failed())
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        startOp, stream_emulator_run, mlir::TypeRange{},
        startOp.getOperation()->getOperands());
    return ::mlir::success();
  };
};

struct LowerSDFGShutdown
    : public mlir::OpRewritePattern<mlir::concretelang::SDFG::Shutdown> {
  LowerSDFGShutdown(::mlir::MLIRContext *context,
                    mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::SDFG::Shutdown>(context,
                                                                     benefit) {}
  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::SDFG::Shutdown desOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    mlir::FunctionType funcType = mlir::FunctionType::get(
        rewriter.getContext(), {SDFG::DFGType::get(rewriter.getContext())}, {});
    if (insertForwardDeclaration(desOp, rewriter, stream_emulator_delete,
                                 funcType)
            .failed())
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        desOp, stream_emulator_delete, mlir::TypeRange{},
        desOp.getOperation()->getOperands());
    return ::mlir::success();
  };
};

struct LowerSDFGMakeProcess
    : public mlir::OpRewritePattern<mlir::concretelang::SDFG::MakeProcess> {
  LowerSDFGMakeProcess(::mlir::MLIRContext *context,
                       mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::SDFG::MakeProcess>(
            context, benefit) {}
  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::SDFG::MakeProcess mpOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    const char *funcName = nullptr;
    mlir::SmallVector<mlir::Value> operands(mpOp->getOperands());
    switch (mpOp.getType()) {
    case SDFG::ProcessKind::add_eint:
      funcName = stream_emulator_make_memref_add_lwe_ciphertexts_u64_process;
      break;
    case SDFG::ProcessKind::add_eint_int:
      funcName =
          stream_emulator_make_memref_add_plaintext_lwe_ciphertext_u64_process;
      break;
    case SDFG::ProcessKind::mul_eint_int:
      funcName =
          stream_emulator_make_memref_mul_cleartext_lwe_ciphertext_u64_process;
      break;
    case SDFG::ProcessKind::neg_eint:
      funcName = stream_emulator_make_memref_negate_lwe_ciphertext_u64_process;
      break;
    case SDFG::ProcessKind::batched_keyswitch:
      funcName = stream_emulator_make_memref_batched_keyswitch_lwe_u64_process;
      [[fallthrough]];
    case SDFG::ProcessKind::keyswitch:
      if (funcName == nullptr)
        funcName = stream_emulator_make_memref_keyswitch_lwe_u64_process;
      // level
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(), mpOp->getAttrOfType<mlir::IntegerAttr>("level")));
      // base_log
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(), mpOp->getAttrOfType<mlir::IntegerAttr>("baseLog")));
      // lwe_dim_in
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(), mpOp->getAttrOfType<mlir::IntegerAttr>("lwe_dim_in")));
      // lwe_dim_out
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(),
          mpOp->getAttrOfType<mlir::IntegerAttr>("lwe_dim_out")));
      // output_size
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(),
          mpOp->getAttrOfType<mlir::IntegerAttr>("output_size")));
      // ksk_index
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(), mpOp->getAttrOfType<mlir::IntegerAttr>("kskIndex")));
      // context
      operands.push_back(getContextArgument(mpOp));
      break;
    case SDFG::ProcessKind::batched_bootstrap:
      funcName = stream_emulator_make_memref_batched_bootstrap_lwe_u64_process;
      [[fallthrough]];
    case SDFG::ProcessKind::batched_mapped_bootstrap:
      if (funcName == nullptr)
        funcName =
            stream_emulator_make_memref_batched_mapped_bootstrap_lwe_u64_process;
      [[fallthrough]];
    case SDFG::ProcessKind::bootstrap:
      if (funcName == nullptr)
        funcName = stream_emulator_make_memref_bootstrap_lwe_u64_process;
      // input_lwe_dim
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(),
          mpOp->getAttrOfType<mlir::IntegerAttr>("inputLweDim")));
      // poly_size
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(), mpOp->getAttrOfType<mlir::IntegerAttr>("polySize")));
      // level
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(), mpOp->getAttrOfType<mlir::IntegerAttr>("level")));
      // base_log
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(), mpOp->getAttrOfType<mlir::IntegerAttr>("baseLog")));
      // glwe_dim
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(),
          mpOp->getAttrOfType<mlir::IntegerAttr>("glweDimension")));
      // output_size
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(),
          mpOp->getAttrOfType<mlir::IntegerAttr>("output_size")));
      // bsk_index
      operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
          mpOp.getLoc(), mpOp->getAttrOfType<mlir::IntegerAttr>("bskIndex")));
      // context
      operands.push_back(getContextArgument(mpOp));
      break;
    case SDFG::ProcessKind::batched_add_eint:
      funcName =
          stream_emulator_make_memref_batched_add_lwe_ciphertexts_u64_process;
      break;
    case SDFG::ProcessKind::batched_add_eint_int:
      funcName =
          stream_emulator_make_memref_batched_add_plaintext_lwe_ciphertext_u64_process;
      break;
    case SDFG::ProcessKind::batched_add_eint_int_cst:
      funcName =
          stream_emulator_make_memref_batched_add_plaintext_cst_lwe_ciphertext_u64_process;
      break;
    case SDFG::ProcessKind::batched_mul_eint_int:
      funcName =
          stream_emulator_make_memref_batched_mul_cleartext_lwe_ciphertext_u64_process;
      break;
    case SDFG::ProcessKind::batched_mul_eint_int_cst:
      funcName =
          stream_emulator_make_memref_batched_mul_cleartext_cst_lwe_ciphertext_u64_process;
      break;
    case SDFG::ProcessKind::batched_neg_eint:
      funcName =
          stream_emulator_make_memref_batched_negate_lwe_ciphertext_u64_process;
      break;
    }
    if (insertGenericForwardDeclaration(mpOp, rewriter, funcName,
                                        mlir::ValueRange{operands}.getTypes(),
                                        mpOp->getResultTypes())
            .failed())
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        mpOp, funcName, mpOp->getResultTypes(), operands);
    return ::mlir::success();
  };
};

struct LowerSDFGMakeStream
    : public mlir::OpRewritePattern<mlir::concretelang::SDFG::MakeStream> {
  LowerSDFGMakeStream(::mlir::MLIRContext *context,
                      mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::SDFG::MakeStream>(
            context, benefit) {}
  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::SDFG::MakeStream msOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    const char *funcName;

    stream_type t;
    switch (msOp.getType()) {
    case SDFG::StreamKind::host_to_device:
      t = TS_STREAM_TYPE_X86_TO_TOPO_LSAP;
      break;
    case SDFG::StreamKind::on_device:
      t = TS_STREAM_TYPE_TOPO_TO_TOPO_LSAP;
      break;
    case SDFG::StreamKind::device_to_host:
      t = TS_STREAM_TYPE_TOPO_TO_X86_LSAP;
      break;
    case SDFG::StreamKind::device_to_both:
      t = TS_STREAM_TYPE_TOPO_TO_BOTH;
      break;
    }
    auto sType = msOp->getResultTypes()[0].dyn_cast_or_null<SDFG::StreamType>();
    assert(sType && "SDFG MakeStream operation should return a stream type");

    if (sType.getElementType().isa<mlir::RankedTensorType>()) {
      if (sType.getElementType().dyn_cast<mlir::TensorType>().getRank() == 1)
        funcName = stream_emulator_make_memref_stream;
      else if (sType.getElementType().dyn_cast<mlir::TensorType>().getRank() ==
               2)
        funcName = stream_emulator_make_memref_batch_stream;
      else
        return ::mlir::failure();
    } else {
      assert(sType.getElementType().isa<mlir::IntegerType>() &&
             "SDFG streams only support memrefs and integers.");
      funcName = stream_emulator_make_uint64_stream;
    }
    if (insertGenericForwardDeclaration(
            msOp, rewriter, funcName,
            {rewriter.getI64Type(), rewriter.getI64Type()},
            msOp->getResultTypes())
            .failed())
      return ::mlir::failure();
    mlir::Value nullStringPtr = rewriter.create<mlir::arith::ConstantOp>(
        msOp.getLoc(), rewriter.getI64IntegerAttr(0));
    mlir::Value streamTypeCst = rewriter.create<mlir::arith::ConstantOp>(
        msOp.getLoc(), rewriter.getI64IntegerAttr((int)t));
    auto callop = rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        msOp, funcName,
        makeDynamicTensorTypes(rewriter, msOp->getResultTypes()[0]),
        mlir::ValueRange{nullStringPtr, streamTypeCst});
    for (auto &use : llvm::make_early_inc_range(msOp->getUses()))
      use.set(callop.getResult(0));
    return ::mlir::success();
  };
};

struct LowerSDFGPut
    : public mlir::OpRewritePattern<mlir::concretelang::SDFG::Put> {
  LowerSDFGPut(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::SDFG::Put>(context,
                                                                benefit) {}
  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::SDFG::Put putOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    const char *funcName;
    auto sType =
        putOp->getOperandTypes()[0].dyn_cast_or_null<SDFG::StreamType>();
    assert(sType &&
           "SDFG Put operation must take a stream type as first parameter.");
    if (sType.getElementType().isa<mlir::RankedTensorType>()) {
      if (sType.getElementType().dyn_cast<mlir::TensorType>().getRank() == 1)
        funcName = stream_emulator_put_memref;
      else if (sType.getElementType().dyn_cast<mlir::TensorType>().getRank() ==
               2)
        funcName = stream_emulator_put_memref_batch;
      else
        return ::mlir::failure();
    } else {
      assert(sType.getElementType().isa<mlir::IntegerType>() &&
             "SDFG streams only support memrefs and integers.");
      funcName = stream_emulator_put_uint64;
    }
    // Add data ownership flag - if the put operation takes ownership
    // of the memref data, set to 0 by default.
    mlir::SmallVector<mlir::Value> operands(putOp->getOperands());
    operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
        putOp.getLoc(), rewriter.getI64IntegerAttr(0)));

    if (insertGenericForwardDeclaration(putOp, rewriter, funcName,
                                        mlir::ValueRange{operands}.getTypes(),
                                        putOp->getResultTypes())
            .failed())
      return ::mlir::failure();
    mlir::SmallVector<mlir::Value> newOps;
    castDynamicTensorOps(putOp, rewriter, operands, newOps);
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        putOp, funcName, putOp->getResultTypes(), newOps);
    return ::mlir::success();
  };
};

struct LowerSDFGGet
    : public mlir::OpRewritePattern<mlir::concretelang::SDFG::Get> {
  LowerSDFGGet(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::SDFG::Get>(context,
                                                                benefit) {}
  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::SDFG::Get getOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    const char *funcName;
    auto sType =
        getOp->getOperandTypes()[0].dyn_cast_or_null<SDFG::StreamType>();
    assert(sType &&
           "SDFG Get operation must take a stream type as first parameter.");
    if (sType.getElementType().isa<mlir::RankedTensorType>()) {
      // TODO: SDFG.Get for memref streams is lowered during bufferization
      // as returning a memref requires allocation for now
      return ::mlir::success();
    } else {
      assert(sType.getElementType().isa<mlir::IntegerType>() &&
             "SDFG streams only support memrefs and integers.");
      funcName = stream_emulator_get_uint64;
    }
    if (insertGenericForwardDeclaration(getOp, rewriter, funcName,
                                        getOp->getOperandTypes(),
                                        getOp->getResultTypes())
            .failed())
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        getOp, funcName, getOp->getResultTypes(), getOp->getOperands());
    return ::mlir::success();
  };
};
} // namespace

void SDFGToStreamEmulatorPass::runOnOperation() {
  auto op = this->getOperation();
  mlir::ConversionTarget target(getContext());
  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<LowerSDFGInit, LowerSDFGStart, LowerSDFGShutdown,
                  LowerSDFGMakeProcess, LowerSDFGMakeStream, LowerSDFGPut,
                  LowerSDFGGet>(&getContext());

  target.addIllegalOp<SDFG::Init, SDFG::Start, SDFG::Shutdown,
                      SDFG::MakeProcess, SDFG::MakeStream, SDFG::Put>();
  // All Concrete ops are legal after the conversion
  target.addLegalDialect<mlir::concretelang::Concrete::ConcreteDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalOp<mlir::func::ReturnOp, mlir::func::FuncOp,
                    mlir::func::CallOp, SDFG::Get, mlir::tensor::CastOp>();

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertSDFGToStreamEmulatorPass() {
  return std::make_unique<SDFGToStreamEmulatorPass>();
}
} // namespace concretelang
} // namespace mlir
