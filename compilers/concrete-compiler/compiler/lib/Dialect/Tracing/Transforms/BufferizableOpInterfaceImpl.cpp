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

#include "concretelang/Dialect/Tracing/IR/TracingDialect.h"
#include "concretelang/Dialect/Tracing/IR/TracingOps.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::concretelang::Tracing;

namespace {

template <typename Op>
struct TrivialBufferizableInterface
    : public BufferizableOpInterface::ExternalModel<
          TrivialBufferizableInterface<Op>, Op> {
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

    mlir::SmallVector<mlir::Value> operands;
    for (auto &operand : op->getOpOperands()) {
      if (!operand.get().getType().isa<mlir::RankedTensorType>()) {
        operands.push_back(operand.get());
      } else {
        operands.push_back(
            *bufferization::getBuffer(rewriter, operand.get(), options));
      }
    }

    rewriter.replaceOpWithNewOp<Op>(op, mlir::TypeRange{}, operands,
                                    op->getAttrs());

    return success();
  }
};

} // namespace

namespace mlir {
namespace concretelang {
namespace Tracing {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TracingDialect *dialect) {
    // trace_ciphretext
    Tracing::TraceCiphertextOp::attachInterface<
        TrivialBufferizableInterface<Tracing::TraceCiphertextOp>>(*ctx);
    // trace_plaintext
    Tracing::TracePlaintextOp::attachInterface<
        TrivialBufferizableInterface<Tracing::TracePlaintextOp>>(*ctx);
    // trace_message
    Tracing::TraceMessageOp::attachInterface<
        TrivialBufferizableInterface<Tracing::TraceMessageOp>>(*ctx);
  });
}
} // namespace Tracing
} // namespace concretelang
} // namespace mlir
