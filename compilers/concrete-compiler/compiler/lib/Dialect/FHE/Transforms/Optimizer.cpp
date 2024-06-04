// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <concretelang/Dialect/FHE/IR/FHEOps.h>
#include <concretelang/Dialect/FHE/Transforms/Optimizer/Optimizer.h>
#include <concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h>

#include <llvm/Support/Debug.h>

namespace mlir {
namespace concretelang {

struct OptimizerPartitionFrontierMaterializationPass
    : public OptimizerPartitionFrontierMaterializationPassBase<
          OptimizerPartitionFrontierMaterializationPass> {

  OptimizerPartitionFrontierMaterializationPass(
      const optimizer::CircuitSolution &solverSolution)
      : solverSolution(solverSolution) {}

  enum class OperationKind { PRODUCER, CONSUMER };

  std::optional<uint64_t> getOid(mlir::Operation *op, OperationKind kind) {
    if (mlir::IntegerAttr oidAttr =
            op->getAttrOfType<mlir::IntegerAttr>("TFHE.OId")) {
      return oidAttr.getInt();
    } else if (mlir::DenseI32ArrayAttr oidArrayAttr =
                   op->getAttrOfType<mlir::DenseI32ArrayAttr>("TFHE.OId")) {
      assert(oidArrayAttr.size() > 0);

      if (kind == OperationKind::CONSUMER) {
        return oidArrayAttr[0];
      } else {
        // All operations with a `TFHE.OId` array attribute store the
        // OId of the result at the last position, except
        // multiplications, which use the 6th element (at index 5),
        // see `mlir::concretelang::optimizer::FunctionToDag::addMul`.
        if (llvm::dyn_cast<mlir::concretelang::FHE::MulEintOp>(op) ||
            llvm::dyn_cast<mlir::concretelang::FHELinalg::MulEintOp>(op)) {
          assert(oidArrayAttr.size() >= 6);
          return oidArrayAttr[5];
        } else {
          return oidArrayAttr[oidArrayAttr.size() - 1];
        }
      }
    } else {
      return std::nullopt;
    }
  }

  void runOnOperation() final {
    mlir::func::FuncOp func = this->getOperation();

    func.walk([&](mlir::Operation *producer) {
      mlir::IRRewriter rewriter(producer->getContext());

      // Remove the change_partition op.
      // TODO: The crypto parameters used in the op should be considered before
      // removal
      if (mlir::dyn_cast_or_null<FHELinalg::ChangePartitionEintOp>(producer) ||
          mlir::dyn_cast_or_null<FHE::ChangePartitionEintOp>(producer)) {
        rewriter.startRootUpdate(func);
        rewriter.replaceOp(producer, producer->getOperand(0));
        rewriter.finalizeRootUpdate(func);
        return;
      }

      std::optional<uint64_t> producerOid =
          getOid(producer, OperationKind::PRODUCER);

      if (!producerOid.has_value())
        return;

      assert(*producerOid < solverSolution.instructions_keys.size());

      auto &eck =
          solverSolution.instructions_keys[*producerOid].extra_conversion_keys;

      if (eck.size() == 0)
        return;

      assert(eck.size() == 1);
      assert(eck[0] <
             solverSolution.circuit_keys.conversion_keyswitch_keys.size());

      uint64_t producerOutKeyID =
          solverSolution.instructions_keys[*producerOid].output_key;

      uint64_t conversionOutKeyID =
          solverSolution.circuit_keys.conversion_keyswitch_keys[eck[0]]
              .output_key.identifier;

      rewriter.setInsertionPointAfter(producer);

      for (mlir::Value res : producer->getResults()) {
        mlir::Value resConverted;

        for (mlir::OpOperand &operand :
             llvm::make_early_inc_range(res.getUses())) {
          mlir::Operation *consumer = operand.getOwner();

          std::optional<uint64_t> consumerOid =
              getOid(consumer, OperationKind::CONSUMER);

          // By default, all consumers need the converted value,
          // unless it is explicitly specified that the original value
          // is needed
          bool needsConvertedValue = true;

          if (consumerOid.has_value()) {
            assert(*consumerOid < solverSolution.instructions_keys.size());

            uint64_t consumerInKeyID =
                solverSolution.instructions_keys[*consumerOid].input_key;

            if (consumerInKeyID == producerOutKeyID) {
              needsConvertedValue = false;
            } else {
              assert(consumerInKeyID == conversionOutKeyID &&
                     "Consumer needs converted value, but with a key that is "
                     "not the extra conversion key of the producer");
            }
          }

          if (needsConvertedValue) {
            if (!resConverted) {
              resConverted = rewriter.create<Optimizer::PartitionFrontierOp>(
                  producer->getLoc(), res.getType(), res, producerOutKeyID,
                  conversionOutKeyID);
            }

            operand.set(resConverted);
          }
        }
      }
    });
  }

protected:
  const optimizer::CircuitSolution &solverSolution;
};

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createOptimizerPartitionFrontierMaterializationPass(
    const optimizer::CircuitSolution &solverSolution) {
  return std::make_unique<OptimizerPartitionFrontierMaterializationPass>(
      solverSolution);
}

} // namespace concretelang
} // namespace mlir
