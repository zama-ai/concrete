// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "concretelang/Conversion/Tools.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteDialect.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.h"
#include "concretelang/Dialect/BConcrete/Transforms/BufferizableOpInterfaceImpl.h"
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;

namespace BConcrete = mlir::concretelang::BConcrete;

namespace mlir {
namespace concretelang {
namespace BConcrete {
namespace {} // namespace
} // namespace BConcrete
} // namespace concretelang
} // namespace mlir

namespace {

// Returns a map with a symbolic offset for each dimension, i.e., for N
// dimensions, it returns
//
// [d1, d2, ..., dN](s1, s2, ..., sN) -> (d1 + s1, d2 + s2, ..., dN + sN)
//
AffineMap getMultiDimSymbolicOffsetMap(mlir::RewriterBase &rewriter,
                                       unsigned rank) {
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(rank);

  for (unsigned i = 0; i < rank; ++i)
    dimExprs.push_back(rewriter.getAffineDimExpr(i) +
                       rewriter.getAffineSymbolExpr(i));

  return AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/rank, dimExprs,
                        rewriter.getContext());
}

mlir::Type getDynamicMemrefWithUnknownOffset(mlir::RewriterBase &rewriter,
                                             size_t rank) {
  mlir::MLIRContext *ctx = rewriter.getContext();
  std::vector<int64_t> shape(rank, -1);

  return mlir::MemRefType::get(shape, rewriter.getI64Type(),
                               getMultiDimSymbolicOffsetMap(rewriter, rank));
}

// Returns `memref.cast %0 : memref<...xAxT> to memref<...x?xT>`
mlir::Value getCastedMemRef(mlir::RewriterBase &rewriter, mlir::Location loc,
                            mlir::Value value) {
  mlir::Type valueType = value.getType();
  if (auto memrefTy = valueType.dyn_cast_or_null<mlir::MemRefType>()) {
    return rewriter.create<mlir::memref::CastOp>(
        loc,
        getDynamicMemrefWithUnknownOffset(rewriter, memrefTy.getShape().size()),
        value);
  } else {
    return value;
  }
}

char memref_add_lwe_ciphertexts_u64[] = "memref_add_lwe_ciphertexts_u64";
char memref_add_plaintext_lwe_ciphertext_u64[] =
    "memref_add_plaintext_lwe_ciphertext_u64";
char memref_mul_cleartext_lwe_ciphertext_u64[] =
    "memref_mul_cleartext_lwe_ciphertext_u64";
char memref_negate_lwe_ciphertext_u64[] = "memref_negate_lwe_ciphertext_u64";
char memref_keyswitch_lwe_u64[] = "memref_keyswitch_lwe_u64";
char memref_bootstrap_lwe_u64[] = "memref_bootstrap_lwe_u64";
char memref_keyswitch_async_lwe_u64[] = "memref_keyswitch_async_lwe_u64";
char memref_bootstrap_async_lwe_u64[] = "memref_bootstrap_async_lwe_u64";
char memref_await_future[] = "memref_await_future";
char memref_expand_lut_in_trivial_glwe_ct_u64[] =
    "memref_expand_lut_in_trivial_glwe_ct_u64";

char memref_wop_pbs_crt_buffer[] = "memref_wop_pbs_crt_buffer";

mlir::LogicalResult insertForwardDeclarationOfTheCAPI(
    mlir::Operation *op, mlir::RewriterBase &rewriter, char const *funcName) {

  auto memref1DType = getDynamicMemrefWithUnknownOffset(rewriter, 1);
  auto memref2DType = getDynamicMemrefWithUnknownOffset(rewriter, 2);
  auto futureType =
      mlir::concretelang::RT::FutureType::get(rewriter.getIndexType());
  auto contextType =
      mlir::concretelang::Concrete::ContextType::get(rewriter.getContext());
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
  } else if (funcName == memref_keyswitch_lwe_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(), {memref1DType, memref1DType, contextType}, {});
  } else if (funcName == memref_bootstrap_lwe_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {memref1DType, memref1DType, memref1DType, contextType}, {});
  } else if (funcName == memref_keyswitch_async_lwe_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(), {memref1DType, memref1DType, contextType},
        {futureType});
  } else if (funcName == memref_bootstrap_async_lwe_u64) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {memref1DType, memref1DType, memref1DType, contextType}, {futureType});
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
                                           contextType,
                                       },
                                       {});
  } else {
    op->emitError("unknwon external function") << funcName;
    return mlir::failure();
  }

  return insertForwardDeclaration(op, rewriter, funcName, funcType);
}

/// Returns the value of the context argument from the enclosing func
mlir::Value getContextArgument(mlir::Operation *op) {
  mlir::Block *block = op->getBlock();
  while (block != nullptr) {
    if (llvm::isa<mlir::func::FuncOp>(block->getParentOp())) {
      block = &mlir::cast<mlir::func::FuncOp>(block->getParentOp())
                   .getBody()
                   .front();

      auto context =
          std::find_if(block->getArguments().rbegin(),
                       block->getArguments().rend(), [](BlockArgument &arg) {
                         return arg.getType()
                             .isa<mlir::concretelang::Concrete::ContextType>();
                       });
      assert(context != block->getArguments().rend() &&
             "Cannot find the Concrete.context");

      return *context;
    }
    block = block->getParentOp()->getBlock();
  }
  assert("can't find a function that enclose the op");
  return nullptr;
}

template <typename Op, char const *funcName, bool withContext = false>
struct BufferizableWithCallOpInterface
    : public BufferizableOpInterface::ExternalModel<
          BufferizableWithCallOpInterface<Op, funcName, withContext>, Op> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::None;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {

    auto loc = op->getLoc();
    auto castOp = cast<Op>(op);

    // For now we always alloc for the result, we didn't have the in place
    // operators yet.
    auto resTensorType =
        castOp.result().getType().template cast<mlir::TensorType>();

    auto outMemrefType = MemRefType::get(resTensorType.getShape(),
                                         resTensorType.getElementType());
    auto outMemref = options.createAlloc(rewriter, loc, outMemrefType, {});
    if (mlir::failed(outMemref)) {
      return mlir::failure();
    }

    // The first operand is the result
    mlir::SmallVector<mlir::Value, 3> operands{
        getCastedMemRef(rewriter, loc, *outMemref),
    };
    // For all tensor operand get the corresponding casted buffer
    for (auto &operand : op->getOpOperands()) {
      if (!operand.get().getType().isa<mlir::RankedTensorType>()) {
        operands.push_back(operand.get());
      } else {
        auto memrefOperand =
            bufferization::getBuffer(rewriter, operand.get(), options);
        operands.push_back(getCastedMemRef(rewriter, loc, memrefOperand));
      }
    }
    // Append the context argument
    if (withContext) {
      operands.push_back(getContextArgument(op));
    }

    // Insert forward declaration of the function
    if (insertForwardDeclarationOfTheCAPI(op, rewriter, funcName).failed()) {
      return mlir::failure();
    }

    rewriter.create<mlir::func::CallOp>(loc, funcName, mlir::TypeRange{},
                                        operands);

    replaceOpWithBufferizedValues(rewriter, op, *outMemref);

    return success();
  }
};

struct BufferizableGlweFromTableOpInterface
    : public BufferizableOpInterface::ExternalModel<
          BufferizableGlweFromTableOpInterface, BConcrete::FillGlweFromTable> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::None;
  }

  /// Bufferize GlweFromTable
  /// ```
  /// "BConcrete.fill_glwe_table"(%glwe, %lut) {glweDimension=1,
  /// polynomialSize=2048, outPrecision=3} :
  ///   (tensor<4096xi64>, tensor<32xi64>) -> ()
  /// ```
  ///
  /// to
  ///
  /// ```
  /// %glweDim = arith.constant 1 : i32
  /// %polySize = arith.constant 2048 : i32
  /// %outPrecision = arith.constant 3 : i32
  /// %glwe_ = memref.cast %glwe : memref<4096xi64> to memref<?xi64>
  /// %lut_ = memref.cast %lut : memref<32xi64> to memref<?xi64>
  /// call @expand_lut_in_trivial_glwe_ct(%glwe, %polySize, %glweDim,
  /// %outPrecision, %lut_) :
  ///   (tensor<?xi64>, i32, i32, tensor<?xi64>) -> ()
  /// ```
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {

    auto loc = op->getLoc();
    auto castOp = cast<BConcrete::FillGlweFromTable>(op);

    auto glweOp =
        getCastedMemRef(rewriter, loc,
                        bufferization::getBuffer(
                            rewriter, castOp->getOpOperand(0).get(), options));
    auto lutOp =
        getCastedMemRef(rewriter, loc,
                        bufferization::getBuffer(
                            rewriter, castOp->getOpOperand(1).get(), options));

    auto polySizeOp = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(castOp.polynomialSize()));
    auto glweDimensionOp = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(castOp.glweDimension()));
    auto outPrecisionOp = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(castOp.outPrecision()));

    mlir::SmallVector<mlir::Value> operands{glweOp, polySizeOp, glweDimensionOp,
                                            outPrecisionOp, lutOp};

    // Insert forward declaration of the function
    if (insertForwardDeclarationOfTheCAPI(
            op, rewriter, memref_expand_lut_in_trivial_glwe_ct_u64)
            .failed()) {
      return mlir::failure();
    }

    rewriter.create<mlir::func::CallOp>(
        loc, memref_expand_lut_in_trivial_glwe_ct_u64, mlir::TypeRange{},
        operands);

    replaceOpWithBufferizedValues(rewriter, op, {});

    return success();
  }
};

template <typename Op, char const *funcName, bool withContext = false>
struct BufferizableWithAsyncCallOpInterface
    : public BufferizableOpInterface::ExternalModel<
          BufferizableWithAsyncCallOpInterface<Op, funcName, withContext>, Op> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::None;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {

    auto loc = op->getLoc();
    auto castOp = cast<Op>(op);

    // For now we always alloc for the result, we didn't have the in place
    // operators yet.
    auto resTensorType =
        castOp.result()
            .getType()
            .template cast<mlir::concretelang::RT::FutureType>()
            .getElementType()
            .template cast<mlir::TensorType>();

    auto outMemrefType = MemRefType::get(resTensorType.getShape(),
                                         resTensorType.getElementType());
    auto outMemref = options.createAlloc(rewriter, loc, outMemrefType, {});
    if (mlir::failed(outMemref)) {
      return mlir::failure();
    }

    // The first operand is the result
    mlir::SmallVector<mlir::Value, 3> operands{
        getCastedMemRef(rewriter, loc, *outMemref),
    };
    // For all tensor operand get the corresponding casted buffer
    for (auto &operand : op->getOpOperands()) {
      if (!operand.get().getType().isa<mlir::RankedTensorType>()) {
        operands.push_back(operand.get());
      } else {
        auto memrefOperand =
            bufferization::getBuffer(rewriter, operand.get(), options);
        operands.push_back(getCastedMemRef(rewriter, loc, memrefOperand));
      }
    }
    // Append the context argument
    if (withContext) {
      operands.push_back(getContextArgument(op));
    }

    // Insert forward declaration of the function
    if (insertForwardDeclarationOfTheCAPI(op, rewriter, funcName).failed()) {
      return mlir::failure();
    }

    auto result = rewriter.create<mlir::func::CallOp>(
        loc, funcName,
        mlir::TypeRange{
            mlir::concretelang::RT::FutureType::get(rewriter.getIndexType())},
        operands);

    replaceOpWithBufferizedValues(rewriter, op, result.getResult(0));

    return success();
  }
};

template <typename Op, char const *funcName>
struct BufferizableWithSyncCallOpInterface
    : public BufferizableOpInterface::ExternalModel<
          BufferizableWithSyncCallOpInterface<Op, funcName>, Op> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::None;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {

    auto loc = op->getLoc();
    auto castOp = cast<Op>(op);

    auto resTensorType =
        castOp.result().getType().template cast<mlir::TensorType>();

    auto outMemrefType = MemRefType::get(resTensorType.getShape(),
                                         resTensorType.getElementType());
    auto outMemref = options.createAlloc(rewriter, loc, outMemrefType, {});
    if (mlir::failed(outMemref)) {
      return mlir::failure();
    }

    // The first operand is the result
    mlir::SmallVector<mlir::Value, 3> operands{
        getCastedMemRef(rewriter, loc, *outMemref),
    };
    // Then add the future operand
    operands.push_back(op->getOpOperand(0).get());
    // Finally add a dependence on the memref covered by the future to
    // prevent early deallocation
    auto def = op->getOpOperand(0).get().getDefiningOp();
    operands.push_back(def->getOpOperand(0).get());
    operands.push_back(def->getOpOperand(1).get());

    // Insert forward declaration of the function
    if (insertForwardDeclarationOfTheCAPI(op, rewriter, funcName).failed()) {
      return mlir::failure();
    }

    rewriter.create<mlir::func::CallOp>(loc, funcName, mlir::TypeRange{},
                                        operands);

    replaceOpWithBufferizedValues(rewriter, op, *outMemref);

    return success();
  }
};

} // namespace

void mlir::concretelang::BConcrete::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            BConcrete::BConcreteDialect *dialect) {
    BConcrete::AddLweBuffersOp::attachInterface<BufferizableWithCallOpInterface<
        BConcrete::AddLweBuffersOp, memref_add_lwe_ciphertexts_u64>>(*ctx);
    BConcrete::AddPlaintextLweBufferOp::attachInterface<
        BufferizableWithCallOpInterface<
            BConcrete::AddPlaintextLweBufferOp,
            memref_add_plaintext_lwe_ciphertext_u64>>(*ctx);
    BConcrete::MulCleartextLweBufferOp::attachInterface<
        BufferizableWithCallOpInterface<
            BConcrete::MulCleartextLweBufferOp,
            memref_mul_cleartext_lwe_ciphertext_u64>>(*ctx);
    BConcrete::NegateLweBufferOp::attachInterface<
        BufferizableWithCallOpInterface<BConcrete::NegateLweBufferOp,
                                        memref_negate_lwe_ciphertext_u64>>(
        *ctx);
    BConcrete::KeySwitchLweBufferOp::attachInterface<
        BufferizableWithCallOpInterface<BConcrete::KeySwitchLweBufferOp,
                                        memref_keyswitch_lwe_u64, true>>(*ctx);
    BConcrete::BootstrapLweBufferOp::attachInterface<
        BufferizableWithCallOpInterface<BConcrete::BootstrapLweBufferOp,
                                        memref_bootstrap_lwe_u64, true>>(*ctx);
    // TODO(16bits): hack
    BConcrete::WopPBSCRTLweBufferOp::attachInterface<
        BufferizableWithCallOpInterface<BConcrete::WopPBSCRTLweBufferOp,
                                        memref_wop_pbs_crt_buffer, true>>(*ctx);
    BConcrete::KeySwitchLweBufferAsyncOffloadOp::attachInterface<
        BufferizableWithAsyncCallOpInterface<
            BConcrete::KeySwitchLweBufferAsyncOffloadOp,
            memref_keyswitch_async_lwe_u64, true>>(*ctx);
    BConcrete::BootstrapLweBufferAsyncOffloadOp::attachInterface<
        BufferizableWithAsyncCallOpInterface<
            BConcrete::BootstrapLweBufferAsyncOffloadOp,
            memref_bootstrap_async_lwe_u64, true>>(*ctx);
    BConcrete::AwaitFutureOp::attachInterface<
        BufferizableWithSyncCallOpInterface<BConcrete::AwaitFutureOp,
                                            memref_await_future>>(*ctx);
    BConcrete::FillGlweFromTable::attachInterface<
        BufferizableGlweFromTableOpInterface>(*ctx);
  });
}
