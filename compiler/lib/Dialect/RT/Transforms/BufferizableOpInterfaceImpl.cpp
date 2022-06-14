// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/RegionUtils.h>

#include "concretelang/Dialect/RT/IR/RTDialect.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::concretelang::RT;
// using namespace mlir::tensor;

namespace {
struct DataflowTaskOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          DataflowTaskOpBufferizationInterface, DataflowTaskOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
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
                          BufferizationState &state) const {
    DataflowTaskOp taskOp = cast<DataflowTaskOp>(op);

    auto isTensorType = [](Type t) { return t.isa<TensorType>(); };
    bool hasTensorResult = llvm::any_of(taskOp.getResultTypes(), isTensorType);
    bool hasTensorOperand =
        llvm::any_of(taskOp.getOperandTypes(), isTensorType);

    if (!hasTensorResult && !hasTensorOperand)
      return success();

    SmallVector<mlir::Value, 2> newOperands;

    rewriter.setInsertionPoint(taskOp.getBody(), taskOp.getBody()->begin());

    for (OpOperand &opOperand : op->getOpOperands()) {
      Value oldOperandValue = opOperand.get();

      if (oldOperandValue.getType().isa<TensorType>()) {
        FailureOr<Value> bufferOrErr = state.getBuffer(rewriter, opOperand);

        if (failed(bufferOrErr))
          return failure();

        Value buffer = bufferOrErr.getValue();
        newOperands.push_back(buffer);

        Value tensor =
            rewriter.create<bufferization::ToTensorOp>(buffer.getLoc(), buffer);

        replaceAllUsesInRegionWith(oldOperandValue, tensor,
                                   taskOp.getBodyRegion());
      }
    }

    if (hasTensorResult) {
      WalkResult wr = taskOp.walk([&](DataflowYieldOp yield) {
        SmallVector<Value, 2> yieldValues;

        for (OpOperand &yieldOperand : yield.getOperation()->getOpOperands())
          if (yieldOperand.get().getType().isa<TensorType>()) {
            FailureOr<Value> bufferOrErr =
                state.getBuffer(rewriter, yieldOperand);

            if (failed(bufferOrErr))
              return WalkResult::interrupt();

            yieldValues.push_back(bufferOrErr.getValue());
          } else {
            yieldValues.push_back(yieldOperand.get());
          }

        rewriter.setInsertionPointAfter(yield);
        rewriter.replaceOpWithNewOp<DataflowYieldOp>(yield.getOperation(),
                                                     yieldValues);

        return WalkResult::advance();
      });

      if (wr.wasInterrupted())
        return failure();
    }

    SmallVector<mlir::Type, 2> newResultTypes;

    for (OpResult res : op->getResults()) {
      if (TensorType t = res.getType().dyn_cast<TensorType>()) {
        BaseMemRefType memrefType = getMemRefType(t, state.getOptions());
        newResultTypes.push_back(memrefType);
      } else {
        newResultTypes.push_back(res.getType());
      }
    }

    rewriter.setInsertionPoint(taskOp);
    DataflowTaskOp newTaskOp = rewriter.create<DataflowTaskOp>(
        taskOp.getLoc(), newResultTypes, newOperands);

    newTaskOp.getRegion().takeBody(taskOp.getRegion());

    replaceOpWithBufferizedValues(rewriter, op, newTaskOp->getResults());

    return success();
  }
};
} // namespace

namespace mlir {
namespace concretelang {
namespace RT {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, RTDialect *dialect) {
    DataflowTaskOp::attachInterface<DataflowTaskOpBufferizationInterface>(*ctx);
  });
}
} // namespace RT
} // namespace concretelang
} // namespace mlir
