// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/Concrete/Transforms/BufferizableOpInterfaceImpl.h"
#include "concretelang/Dialect/Tracing/IR/TracingOps.h"
#include "concretelang/Support/CompilerEngine.h"
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;

namespace {

namespace Tracing = mlir::concretelang::Tracing;
namespace Concrete = mlir::concretelang::Concrete;

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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Unknown;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {

    auto loc = op->getLoc();
    auto castOp = cast<TensorOp>(op);

    auto resTensorType =
        castOp.getResult().getType().template cast<mlir::TensorType>();

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
            *bufferization::getBuffer(rewriter, operand.get(), options));
      }
    }

    rewriter.create<MemrefOp>(loc, mlir::TypeRange{}, operands, op->getAttrs());

    replaceOpWithBufferizedValues(rewriter, op, *outMemref);

    return success();
  }
};

} // namespace

void mlir::concretelang::Concrete::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            Concrete::ConcreteDialect *dialect) {
    // add_lwe_tensor => add_lwe_buffer
    Concrete::AddLweTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::AddLweTensorOp, Concrete::AddLweBufferOp>>(
        *ctx);
    // add_plaintext_lwe_tensor => add_plaintext_lwe_buffer
    Concrete::AddPlaintextLweTensorOp::attachInterface<TensorToMemrefOp<
        Concrete::AddPlaintextLweTensorOp, Concrete::AddPlaintextLweBufferOp>>(
        *ctx);
    // mul_cleartext_lwe_tensor => mul_cleartext_lwe_buffer
    Concrete::MulCleartextLweTensorOp::attachInterface<TensorToMemrefOp<
        Concrete::MulCleartextLweTensorOp, Concrete::MulCleartextLweBufferOp>>(
        *ctx);
    // negate_cleartext_lwe_tensor => negate_cleartext_lwe_buffer
    Concrete::NegateLweTensorOp::attachInterface<TensorToMemrefOp<
        Concrete::NegateLweTensorOp, Concrete::NegateLweBufferOp>>(*ctx);
    // negate_cleartext_lwe_tensor => negate_cleartext_lwe_buffer
    Concrete::NegateLweTensorOp::attachInterface<TensorToMemrefOp<
        Concrete::NegateLweTensorOp, Concrete::NegateLweBufferOp>>(*ctx);
    // keyswitch_lwe_tensor => keyswitch_lwe_buffer
    Concrete::KeySwitchLweTensorOp::attachInterface<TensorToMemrefOp<
        Concrete::KeySwitchLweTensorOp, Concrete::KeySwitchLweBufferOp>>(*ctx);
    // bootstrap_lwe_tensor => bootstrap_lwe_buffer
    Concrete::BootstrapLweTensorOp::attachInterface<TensorToMemrefOp<
        Concrete::BootstrapLweTensorOp, Concrete::BootstrapLweBufferOp>>(*ctx);

    // batched_add_lwe_tensor => batched_add_lwe_buffer
    Concrete::BatchedAddLweTensorOp::attachInterface<TensorToMemrefOp<
        Concrete::BatchedAddLweTensorOp, Concrete::BatchedAddLweBufferOp>>(
        *ctx);
    // batched_add_plaintext_lwe_tensor => batched_add_plaintext_lwe_buffer
    Concrete::BatchedAddPlaintextLweTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::BatchedAddPlaintextLweTensorOp,
                         Concrete::BatchedAddPlaintextLweBufferOp>>(*ctx);
    // batched_add_plaintext_cst_lwe_tensor =>
    // batched_add_plaintext_cst_lwe_buffer
    Concrete::BatchedAddPlaintextCstLweTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::BatchedAddPlaintextCstLweTensorOp,
                         Concrete::BatchedAddPlaintextCstLweBufferOp>>(*ctx);
    // batched_mul_cleartext_lwe_tensor => batched_mul_cleartext_lwe_buffer
    Concrete::BatchedMulCleartextLweTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::BatchedMulCleartextLweTensorOp,
                         Concrete::BatchedMulCleartextLweBufferOp>>(*ctx);
    // batched_mul_cleartext_cst_lwe_tensor =>
    // batched_mul_cleartext_cst_lwe_buffer
    Concrete::BatchedMulCleartextCstLweTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::BatchedMulCleartextCstLweTensorOp,
                         Concrete::BatchedMulCleartextCstLweBufferOp>>(*ctx);
    // batched_negate_lwe_tensor => batched_negate_lwe_buffer
    Concrete::BatchedNegateLweTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::BatchedNegateLweTensorOp,
                         Concrete::BatchedNegateLweBufferOp>>(*ctx);

    // batched_keyswitch_lwe_tensor => batched_keyswitch_lwe_buffer
    Concrete::BatchedKeySwitchLweTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::BatchedKeySwitchLweTensorOp,
                         Concrete::BatchedKeySwitchLweBufferOp>>(*ctx);
    // batched_bootstrap_lwe_tensor => batched_bootstrap_lwe_buffer
    Concrete::BatchedBootstrapLweTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::BatchedBootstrapLweTensorOp,
                         Concrete::BatchedBootstrapLweBufferOp>>(*ctx);
    // batched_mapped_bootstrap_lwe_tensor =>
    // batched_mapped_bootstrap_lwe_buffer
    Concrete::BatchedMappedBootstrapLweTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::BatchedMappedBootstrapLweTensorOp,
                         Concrete::BatchedMappedBootstrapLweBufferOp>>(*ctx);
    // wop_pbs_crt_lwe_tensor => wop_pbs_crt_lwe_buffer
    Concrete::WopPBSCRTLweTensorOp::attachInterface<TensorToMemrefOp<
        Concrete::WopPBSCRTLweTensorOp, Concrete::WopPBSCRTLweBufferOp>>(*ctx);
    // encode_plaintext_with_crt_tensor => encode_plaintext_with_crt_buffer
    Concrete::EncodePlaintextWithCrtTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::EncodePlaintextWithCrtTensorOp,
                         Concrete::EncodePlaintextWithCrtBufferOp>>(*ctx);
    // encode_expand_lut_for_bootstrap_tensor =>
    // encode_expand_lut_for_bootstrap_buffer
    Concrete::EncodeExpandLutForBootstrapTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::EncodeExpandLutForBootstrapTensorOp,
                         Concrete::EncodeExpandLutForBootstrapBufferOp>>(*ctx);
    // encode_lut_for_crt_woppbs_tensor =>
    // encode_lut_for_crt_woppbs_buffer
    Concrete::EncodeLutForCrtWopPBSTensorOp::attachInterface<
        TensorToMemrefOp<Concrete::EncodeLutForCrtWopPBSTensorOp,
                         Concrete::EncodeLutForCrtWopPBSBufferOp>>(*ctx);
  });
}
