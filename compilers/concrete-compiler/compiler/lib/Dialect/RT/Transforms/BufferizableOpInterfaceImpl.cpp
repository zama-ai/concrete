// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/RegionUtils.h>

#include "concretelang/Dialect/RT/IR/RTDialect.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::concretelang::RT;
// using namespace mlir::tensor;

namespace {
struct DerefWorkFunctionArgumentPtrPlaceholderOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          DerefWorkFunctionArgumentPtrPlaceholderOpBufferizationInterface,
          DerefWorkFunctionArgumentPtrPlaceholderOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
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

  LogicalResult bufferize(Operation *bop, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    DerefWorkFunctionArgumentPtrPlaceholderOp op =
        cast<DerefWorkFunctionArgumentPtrPlaceholderOp>(bop);

    auto isTensorType = [](Type t) { return t.isa<TensorType>(); };
    bool hasTensorResult = llvm::any_of(op->getResultTypes(), isTensorType);
    bool hasTensorOperand = llvm::any_of(op->getOperandTypes(), isTensorType);

    if (!hasTensorResult && !hasTensorOperand)
      return success();

    SmallVector<mlir::Value, 2> newOperands;

    for (OpOperand &opOperand : op->getOpOperands()) {
      Value oldOperandValue = opOperand.get();

      if (oldOperandValue.getType().isa<TensorType>()) {
        FailureOr<Value> bufferOrErr =
            bufferization::getBuffer(rewriter, opOperand.get(), options);

        if (failed(bufferOrErr))
          return failure();

        Value buffer = bufferOrErr.value();
        newOperands.push_back(buffer);
      } else {
        newOperands.push_back(opOperand.get());
      }
    }

    SmallVector<mlir::Type, 2> newResultTypes;

    for (OpResult res : op->getResults()) {
      if (TensorType t = res.getType().dyn_cast<TensorType>()) {
        BaseMemRefType memrefType = getMemRefType(res, options);
        newResultTypes.push_back(memrefType);
      } else {
        newResultTypes.push_back(res.getType());
      }
    }

    rewriter.setInsertionPoint(op);
    DerefWorkFunctionArgumentPtrPlaceholderOp newOp =
        rewriter.create<DerefWorkFunctionArgumentPtrPlaceholderOp>(
            op.getLoc(), newResultTypes, newOperands);

    replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());

    return success();
  }
};

struct MakeReadyFutureOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          MakeReadyFutureOpBufferizationInterface, MakeReadyFutureOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
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

  LogicalResult bufferize(Operation *bop, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    MakeReadyFutureOp op = cast<MakeReadyFutureOp>(bop);

    auto isTensorType = [](Type t) { return t.isa<TensorType>(); };
    bool hasTensorResult = llvm::any_of(op->getResultTypes(), isTensorType);
    bool hasTensorOperand = llvm::any_of(op->getOperandTypes(), isTensorType);

    if (!hasTensorResult && !hasTensorOperand)
      return success();

    SmallVector<mlir::Value, 2> newOperands;

    for (OpOperand &opOperand : op->getOpOperands()) {
      Value oldOperandValue = opOperand.get();

      if (oldOperandValue.getType().isa<TensorType>()) {
        FailureOr<Value> bufferOrErr =
            bufferization::getBuffer(rewriter, opOperand.get(), options);

        if (failed(bufferOrErr))
          return failure();

        Value buffer = bufferOrErr.value();
        newOperands.push_back(buffer);
      } else {
        newOperands.push_back(opOperand.get());
      }
    }

    SmallVector<mlir::Type, 2> newResultTypes;

    for (OpResult res : op->getResults()) {
      if (TensorType t = res.getType().dyn_cast<TensorType>()) {
        BaseMemRefType memrefType = getMemRefType(res, options);
        newResultTypes.push_back(memrefType);
      } else {
        newResultTypes.push_back(res.getType());
      }
    }

    rewriter.setInsertionPoint(op);
    MakeReadyFutureOp newOp = rewriter.create<MakeReadyFutureOp>(
        op.getLoc(), newResultTypes, newOperands);

    replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());

    return success();
  }
};

struct WorkFunctionReturnOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          WorkFunctionReturnOpBufferizationInterface, WorkFunctionReturnOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
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

  LogicalResult bufferize(Operation *bop, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    WorkFunctionReturnOp op = cast<WorkFunctionReturnOp>(bop);

    auto isTensorType = [](Type t) { return t.isa<TensorType>(); };
    bool hasTensorResult = llvm::any_of(op->getResultTypes(), isTensorType);
    bool hasTensorOperand = llvm::any_of(op->getOperandTypes(), isTensorType);

    if (!hasTensorResult && !hasTensorOperand)
      return success();

    SmallVector<mlir::Value, 2> newOperands;

    for (OpOperand &opOperand : op->getOpOperands()) {
      Value oldOperandValue = opOperand.get();

      if (oldOperandValue.getType().isa<TensorType>()) {
        FailureOr<Value> bufferOrErr =
            bufferization::getBuffer(rewriter, opOperand.get(), options);

        if (failed(bufferOrErr))
          return failure();

        Value buffer = bufferOrErr.value();
        newOperands.push_back(buffer);
      } else {
        newOperands.push_back(opOperand.get());
      }
    }

    SmallVector<mlir::Type, 2> newResultTypes;

    for (OpResult res : op->getResults()) {
      if (TensorType t = res.getType().dyn_cast<TensorType>()) {
        BaseMemRefType memrefType = getMemRefType(res, options);
        newResultTypes.push_back(memrefType);
      } else {
        newResultTypes.push_back(res.getType());
      }
    }

    rewriter.setInsertionPoint(op);
    WorkFunctionReturnOp newOp = rewriter.create<WorkFunctionReturnOp>(
        op.getLoc(), newResultTypes, newOperands);

    replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());

    return success();
  }
};

struct AwaitFutureOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          AwaitFutureOpBufferizationInterface, AwaitFutureOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
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

  LogicalResult bufferize(Operation *bop, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    AwaitFutureOp op = cast<AwaitFutureOp>(bop);

    auto isTensorType = [](Type t) { return t.isa<TensorType>(); };
    bool hasTensorResult = llvm::any_of(op->getResultTypes(), isTensorType);
    bool hasTensorOperand = llvm::any_of(op->getOperandTypes(), isTensorType);

    if (!hasTensorResult && !hasTensorOperand)
      return success();

    SmallVector<mlir::Value, 2> newOperands;

    for (OpOperand &opOperand : op->getOpOperands()) {
      Value oldOperandValue = opOperand.get();

      if (oldOperandValue.getType().isa<TensorType>()) {
        FailureOr<Value> bufferOrErr =
            bufferization::getBuffer(rewriter, opOperand.get(), options);

        if (failed(bufferOrErr))
          return failure();

        Value buffer = bufferOrErr.value();
        newOperands.push_back(buffer);
      } else {
        newOperands.push_back(opOperand.get());
      }
    }

    SmallVector<mlir::Type, 2> newResultTypes;

    for (OpResult res : op->getResults()) {
      if (TensorType t = res.getType().dyn_cast<TensorType>()) {
        BaseMemRefType memrefType = getMemRefType(res, options);
        newResultTypes.push_back(memrefType);
      } else {
        newResultTypes.push_back(res.getType());
      }
    }

    rewriter.setInsertionPoint(op);
    AwaitFutureOp newOp = rewriter.create<AwaitFutureOp>(
        op.getLoc(), newResultTypes, newOperands);

    replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());

    return success();
  }
};

} // namespace

namespace mlir {
namespace concretelang {
namespace RT {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, RTDialect *dialect) {
    DerefWorkFunctionArgumentPtrPlaceholderOp::attachInterface<
        DerefWorkFunctionArgumentPtrPlaceholderOpBufferizationInterface>(*ctx);
    AwaitFutureOp::attachInterface<AwaitFutureOpBufferizationInterface>(*ctx);
    MakeReadyFutureOp::attachInterface<MakeReadyFutureOpBufferizationInterface>(
        *ctx);
    WorkFunctionReturnOp::attachInterface<
        WorkFunctionReturnOpBufferizationInterface>(*ctx);
  });
}
} // namespace RT
} // namespace concretelang
} // namespace mlir
