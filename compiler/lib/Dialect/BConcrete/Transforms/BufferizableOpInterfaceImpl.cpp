// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "concretelang/Conversion/Tools.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteDialect.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.h"
#include "concretelang/Dialect/BConcrete/Transforms/BufferizableOpInterfaceImpl.h"
#include "concretelang/Support/CompilerEngine.h"
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;

namespace {

namespace BConcrete = mlir::concretelang::BConcrete;

template <typename TensorOp, typename MemrefOp>
struct TensorToMemrefOp : public BufferizableOpInterface::ExternalModel<
                              TensorToMemrefOp<TensorOp, MemrefOp>, TensorOp> {
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
    auto castOp = cast<TensorOp>(op);

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
        *outMemref,
    };
    for (auto &operand : op->getOpOperands()) {
      if (!operand.get().getType().isa<mlir::RankedTensorType>()) {
        operands.push_back(operand.get());
      } else {
        operands.push_back(
            bufferization::getBuffer(rewriter, operand.get(), options));
      }
    }

    rewriter.create<MemrefOp>(loc, mlir::TypeRange{}, operands, op->getAttrs());

    replaceOpWithBufferizedValues(rewriter, op, *outMemref);

    return success();
  }
};

} // namespace

void mlir::concretelang::BConcrete::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            BConcrete::BConcreteDialect *dialect) {
    // add_lwe_tensor => add_lwe_buffer
    BConcrete::AddLweTensorOp::attachInterface<
        TensorToMemrefOp<BConcrete::AddLweTensorOp, BConcrete::AddLweBufferOp>>(
        *ctx);
    // add_plaintext_lwe_tensor => add_plaintext_lwe_buffer
    BConcrete::AddPlaintextLweTensorOp::attachInterface<
        TensorToMemrefOp<BConcrete::AddPlaintextLweTensorOp,
                         BConcrete::AddPlaintextLweBufferOp>>(*ctx);
    // mul_cleartext_lwe_tensor => mul_cleartext_lwe_buffer
    BConcrete::MulCleartextLweTensorOp::attachInterface<
        TensorToMemrefOp<BConcrete::MulCleartextLweTensorOp,
                         BConcrete::MulCleartextLweBufferOp>>(*ctx);
    // negate_cleartext_lwe_tensor => negate_cleartext_lwe_buffer
    BConcrete::NegateLweTensorOp::attachInterface<TensorToMemrefOp<
        BConcrete::NegateLweTensorOp, BConcrete::NegateLweBufferOp>>(*ctx);
    // negate_cleartext_lwe_tensor => negate_cleartext_lwe_buffer
    BConcrete::NegateLweTensorOp::attachInterface<TensorToMemrefOp<
        BConcrete::NegateLweTensorOp, BConcrete::NegateLweBufferOp>>(*ctx);
    // keyswitch_lwe_tensor => keyswitch_lwe_buffer
    BConcrete::KeySwitchLweTensorOp::attachInterface<TensorToMemrefOp<
        BConcrete::KeySwitchLweTensorOp, BConcrete::KeySwitchLweBufferOp>>(
        *ctx);
    // bootstrap_lwe_tensor => bootstrap_lwe_buffer
    BConcrete::BootstrapLweTensorOp::attachInterface<TensorToMemrefOp<
        BConcrete::BootstrapLweTensorOp, BConcrete::BootstrapLweBufferOp>>(
        *ctx);
    // batched_keyswitch_lwe_tensor => batched_keyswitch_lwe_buffer
    BConcrete::BatchedKeySwitchLweTensorOp::attachInterface<
        TensorToMemrefOp<BConcrete::BatchedKeySwitchLweTensorOp,
                         BConcrete::BatchedKeySwitchLweBufferOp>>(*ctx);
    // batched_bootstrap_lwe_tensor => batched_bootstrap_lwe_buffer
    BConcrete::BatchedBootstrapLweTensorOp::attachInterface<
        TensorToMemrefOp<BConcrete::BatchedBootstrapLweTensorOp,
                         BConcrete::BatchedBootstrapLweBufferOp>>(*ctx);
    // wop_pbs_crt_lwe_tensor => wop_pbs_crt_lwe_buffer
    BConcrete::WopPBSCRTLweTensorOp::attachInterface<TensorToMemrefOp<
        BConcrete::WopPBSCRTLweTensorOp, BConcrete::WopPBSCRTLweBufferOp>>(
        *ctx);
  });
}
