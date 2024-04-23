// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Analysis/StaticLoops.h>
#include <concretelang/Dialect/RT/IR/RTDialect.h>
#include <concretelang/Dialect/RT/IR/RTOps.h>
#include <concretelang/Dialect/RT/Transforms/Passes.h>

#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Transforms/RegionUtils.h>

#include <iterator>
#include <optional>

namespace {
bool isClonableIVExpression(mlir::Value v, mlir::scf::ForallOp forallOp);

// Checks if the operation `op` can be cloned safely for insertion
// into a new loop. That is, it must be above `forallOp` or if all of
// its operands only reference loop IVs from `forallOp`, values
// defined above `forallOp` or intermediate values within the body of
// `forallOp` with the same properties. Operations with regions are
// currently not supported.
bool isClonableIVOp(mlir::Operation *op, mlir::scf::ForallOp forallOp) {
  return op->getParentRegion()->isAncestor(&forallOp.getRegion()) ||
         (mlir::isPure(op) && op->getNumRegions() == 0 &&
          llvm::all_of(op->getOperands(), [=](mlir::Value operand) {
            return isClonableIVExpression(operand, forallOp);
          }));
}

// Checks if a value `v` is a loop IV, a value defined above
// `forallOp` or if the defining operation fulfills the conditions of
// `isClonableIVOp`.
bool isClonableIVExpression(mlir::Value v, mlir::scf::ForallOp forallOp) {
  if (llvm::any_of(forallOp.getInductionVars(),
                   [=](mlir::Value iv) { return v == iv; }))
    return true;

  if (mlir::areValuesDefinedAbove(mlir::ValueRange{v},
                                  forallOp.getBodyRegion()))
    return true;

  if (v.getDefiningOp())
    return isClonableIVOp(v.getDefiningOp(), forallOp);

  return false;
}

mlir::Value cloneIVExpression(mlir::IRRewriter &rewriter, mlir::Value v,
                              mlir::IRMapping &mapping,
                              mlir::scf::ForallOp forallOp);

// Clones an operation `op` for insertion into a new loop
mlir::Operation *cloneIVOp(mlir::IRRewriter &rewriter, mlir::Operation *op,
                           mlir::IRMapping &mapping,
                           mlir::scf::ForallOp forallOp) {
  assert(mlir::isPure(op));

  for (mlir::Value operand : op->getOperands()) {
    if (!mapping.contains(operand) &&
        !mlir::areValuesDefinedAbove(mlir::ValueRange{operand},
                                     forallOp.getBodyRegion())) {
      cloneIVExpression(rewriter, operand, mapping, forallOp);
    }
  }

  return rewriter.cloneWithoutRegions(*op, mapping);
}

// If `v` can be referenced safely from a new loop, `v` is returned
// directly. If not, its defining ops are recursively cloned.
mlir::Value cloneIVExpression(mlir::IRRewriter &rewriter, mlir::Value v,
                              mlir::IRMapping &mapping,
                              mlir::scf::ForallOp forallOp) {
  if (mapping.contains(v))
    return mapping.lookup(v);

  if (mlir::areValuesDefinedAbove(mlir::ValueRange{v},
                                  forallOp.getBodyRegion())) {
    return v;
  }

  mlir::Operation *definingOp = v.getDefiningOp();

  assert(definingOp);

  mlir::Operation *clonedOp =
      cloneIVOp(rewriter, definingOp, mapping, forallOp);

  for (auto [res, cloneRes] :
       llvm::zip_equal(definingOp->getResults(), clonedOp->getResults())) {
    mapping.map(res, cloneRes);
  }

  return mapping.lookup(v);
}

struct HoistAwaitFuturePass
    : public HoistAwaitFuturePassBase<HoistAwaitFuturePass> {
  // Checks if all values of `a` are sizes of a non-dynamic dimensions
  bool allStatic(llvm::ArrayRef<int64_t> a) {
    return llvm::all_of(
        a, [](int64_t r) { return !mlir::ShapedType::isDynamic(r); });
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    llvm::SmallVector<mlir::Operation *> opsToErase;

    func.walk([&](mlir::concretelang::RT::AwaitFutureOp awaitFutureOp) {
      // Make sure there are no other consumers that rely on the
      // synchronization
      if (!awaitFutureOp.getResult().hasOneUse())
        return;

      mlir::scf::ForallOp forallOp =
          llvm::dyn_cast<mlir::scf::ForallOp>(awaitFutureOp->getParentOp());

      if (!forallOp)
        return;

      mlir::tensor::ParallelInsertSliceOp parallelInsertSliceOp =
          llvm::dyn_cast<mlir::tensor::ParallelInsertSliceOp>(
              awaitFutureOp.getResult().getUses().begin()->getOwner());

      if (!parallelInsertSliceOp)
        return;

      // Make sure that all indexes, offsets and strides used by the
      // parallel insert slice op depend only on IVs of the forall, on
      // intermediate values produced in the body or on values defined
      // above.
      auto isAttrOrClonableIVExpression = [=](mlir::OpFoldResult ofr) {
        return ofr.is<mlir::Attribute>() ||
               isClonableIVExpression(ofr.dyn_cast<mlir::Value>(), forallOp);
      };

      if (!llvm::all_of(parallelInsertSliceOp.getMixedOffsets(),
                        isAttrOrClonableIVExpression) ||
          !llvm::all_of(parallelInsertSliceOp.getMixedStrides(),
                        isAttrOrClonableIVExpression) ||
          !llvm::all_of(parallelInsertSliceOp.getMixedSizes(),
                        isAttrOrClonableIVExpression)) {
        return;
      }

      // Make sure that the original tensor into which the
      // synchronized values are inserted is a region out argument of
      // the forall op and thus being written to concurrently
      mlir::Value dst = parallelInsertSliceOp.getDest();

      if (!llvm::any_of(forallOp.getRegionOutArgs(),
                        [=](mlir::Value output) { return output == dst; }))
        return;

      // Currently, the tensor storing the futures must have a static
      // shape, so only loops with static trip counts are supported
      if (!(allStatic(forallOp.getStaticLowerBound()) &&
            allStatic(forallOp.getStaticUpperBound()) &&
            allStatic(forallOp.getStaticStep())))
        return;

      llvm::SmallVector<int64_t> tripCounts;

      for (auto [lb, ub, step] : llvm::zip_equal(forallOp.getStaticLowerBound(),
                                                 forallOp.getStaticUpperBound(),
                                                 forallOp.getStaticStep())) {
        tripCounts.push_back(
            mlir::concretelang::getStaticTripCount(lb, ub, step));
      }

      mlir::IRRewriter rewriter(&getContext());
      rewriter.setInsertionPoint(forallOp);

      mlir::Value tensorOfFutures = rewriter.create<mlir::tensor::EmptyOp>(
          forallOp.getLoc(), tripCounts, awaitFutureOp.getInput().getType());

      // Assemble the list of shared outputs that are to be preserved
      // after the output storing the results of the `RT.await_future`
      // has been removed
      llvm::SmallVector<mlir::Value> newOutputs;
      mlir::Value tensorOfValues;
      size_t i = 0;
      size_t oldResultIdx;
      for (auto [output, regionOutArg] : llvm::zip_equal(
               forallOp.getOutputs(), forallOp.getRegionOutArgs())) {
        if (regionOutArg != dst) {
          newOutputs.push_back(output);
        } else {
          tensorOfValues = output;
          oldResultIdx = i;
        }

        i++;
      }

      newOutputs.push_back(tensorOfFutures);

      // Create a new forall loop with the same shared outputs except
      // for the one previously storing the contents of the
      // `RT.await_future` ops is replaced with a tensor of futures
      rewriter.setInsertionPointAfter(forallOp);
      mlir::scf::ForallOp newForallOp = rewriter.create<mlir::scf::ForallOp>(
          forallOp.getLoc(), forallOp.getMixedLowerBound(),
          forallOp.getMixedUpperBound(), forallOp.getMixedStep(), newOutputs,
          std::nullopt);

      // Move all operations from the old forall op to the new one
      auto &newOperations = newForallOp.getBody()->getOperations();
      mlir::Block *oldBody = forallOp.getBody();

      newOperations.splice(newOperations.begin(), oldBody->getOperations(),
                           oldBody->begin(), std::prev(oldBody->end()));

      // Wrap future in a tensor of one element, so that it can be
      // stored in the new shared output tensor of futures using
      // `tensor.parallel_insert_slice`
      rewriter.setInsertionPointAfter(awaitFutureOp);
      mlir::Value futureAsTensor =
          rewriter.create<mlir::tensor::FromElementsOp>(
              awaitFutureOp.getLoc(),
              mlir::ValueRange{awaitFutureOp.getInput()});

      // Move all operations from the old `scf.forall.in_parallel`
      // terminator to the new one
      mlir::scf::InParallelOp oldTerminator = forallOp.getTerminator();
      mlir::scf::InParallelOp newTerminator = newForallOp.getTerminator();

      mlir::Block::OpListType &oldTerminatorOps =
          oldTerminator.getRegion().getBlocks().begin()->getOperations();
      mlir::Block::OpListType &newTerminatorOps =
          newTerminator.getRegion().getBlocks().begin()->getOperations();

      newTerminatorOps.splice(newTerminatorOps.begin(), oldTerminatorOps,
                              oldTerminatorOps.begin(), oldTerminatorOps.end());

      // Remap IVs and out args
      for (auto [oldIV, newIV] : llvm::zip(forallOp.getInductionVars(),
                                           newForallOp.getInductionVars())) {
        oldIV.replaceAllUsesWith(newIV);
      }

      {
        size_t offs = 0;
        for (auto it : llvm::enumerate(forallOp.getRegionOutArgs())) {
          mlir::Value oldRegionOutArg = it.value();

          if (oldRegionOutArg != dst) {
            oldRegionOutArg.replaceAllUsesWith(
                newForallOp.getRegionOutArgs()[it.index() - offs]);
          } else {
            offs++;
          }
        }
      }

      // Create new `tensor.parallel_inset_slice` operation inserting
      // the future into the tensor of futures
      llvm::SmallVector<mlir::OpFoldResult> ones(tripCounts.size(),
                                                 rewriter.getI64IntegerAttr(1));

      mlir::Value tensorOfFuturesRegionOutArg =
          newForallOp.getRegionOutArgs().back();

      mlir::ImplicitLocOpBuilder ilob(parallelInsertSliceOp.getLoc(), rewriter);

      rewriter.setInsertionPointAfter(parallelInsertSliceOp);
      rewriter.create<mlir::tensor::ParallelInsertSliceOp>(
          parallelInsertSliceOp.getLoc(), futureAsTensor,
          tensorOfFuturesRegionOutArg,
          mlir::getAsOpFoldResult(mlir::concretelang::normalizeInductionVars(
              ilob, newForallOp.getInductionVars(),
              newForallOp.getMixedLowerBound(), newForallOp.getMixedStep())),
          ones, ones);

      // Create a new forall loop, that invokes `RT.await_future` on
      // all futures stored in the tensor of futures and writes the
      // contents into the otiginal tensor with the results
      rewriter.setInsertionPointAfter(newForallOp);
      mlir::scf::ForallOp syncForallOp = rewriter.create<mlir::scf::ForallOp>(
          forallOp.getLoc(), forallOp.getMixedLowerBound(),
          forallOp.getMixedUpperBound(), forallOp.getMixedStep(),
          mlir::ValueRange{tensorOfValues}, std::nullopt);

      mlir::Value resultTensorOfFutures = newForallOp.getResults().back();

      rewriter.setInsertionPointToStart(syncForallOp.getBody());
      mlir::Value extractedFuture = rewriter.create<mlir::tensor::ExtractOp>(
          awaitFutureOp.getLoc(), resultTensorOfFutures,
          syncForallOp.getInductionVars());
      mlir::concretelang::RT::AwaitFutureOp newAwaitFutureOp =
          rewriter.create<mlir::concretelang::RT::AwaitFutureOp>(
              awaitFutureOp.getLoc(), awaitFutureOp.getResult().getType(),
              extractedFuture);

      mlir::IRMapping syncMapping;

      for (auto [oldIV, newIV] :
           llvm::zip_equal(newForallOp.getInductionVars(),
                           syncForallOp.getInductionVars())) {
        syncMapping.map(oldIV, newIV);
      }

      syncMapping.map(dst, syncForallOp.getOutputBlockArguments().back());
      syncMapping.map(parallelInsertSliceOp.getSource(),
                      newAwaitFutureOp.getResult());

      auto addMapping = [&](llvm::ArrayRef<mlir::OpFoldResult> ofrs) {
        for (mlir::OpFoldResult ofr : ofrs) {
          if (mlir::Value v = ofr.dyn_cast<mlir::Value>())
            syncMapping.map(
                v, cloneIVExpression(rewriter, v, syncMapping, forallOp));
        }
      };

      addMapping(parallelInsertSliceOp.getMixedOffsets());
      addMapping(parallelInsertSliceOp.getMixedStrides());
      addMapping(parallelInsertSliceOp.getMixedSizes());

      mlir::scf::InParallelOp syncTerminator = syncForallOp.getTerminator();
      rewriter.setInsertionPointToStart(syncTerminator.getBody());
      rewriter.clone(*parallelInsertSliceOp.getOperation(), syncMapping);

      // Replace uses of the results of the original forall loop with:
      // either the corresponding result from the new forall loop if
      // this is a result unrelated to the futures or with the result
      // of the forall loop synchronizing the futures
      {
        size_t offs = 0;
        for (size_t i = 0; i < forallOp.getNumResults(); i++) {
          if (i == oldResultIdx) {
            forallOp.getResult(i).replaceAllUsesWith(syncForallOp.getResult(0));
            offs = 1;
          } else {
            forallOp.getResult(i).replaceAllUsesWith(
                newForallOp.getResult(i - offs));
          }
        }
      }

      // Replace the use of the shared output with the results of the
      // original forall loop with the tensor outside of the loop so
      // that there are no more references to values that were local
      // to the original forall loop, enabling safe erasing of the old
      // operations within the original forall loop
      dst.replaceAllUsesWith(
          forallOp.getOutputs().drop_front(oldResultIdx).front());
      parallelInsertSliceOp->erase();
      awaitFutureOp.erase();

      // Defer erasing the original parallel loop that contained the
      // `RT.await_future` operation until later in order to not
      // confuse the walk relying on the parent operation
      opsToErase.push_back(forallOp);
    });

    for (mlir::Operation *op : opsToErase)
      op->erase();
  }
};
} // namespace

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<func::FuncOp>> createHoistAwaitFuturePass() {
  return std::make_unique<HoistAwaitFuturePass>();
}
} // namespace concretelang
} // namespace mlir
