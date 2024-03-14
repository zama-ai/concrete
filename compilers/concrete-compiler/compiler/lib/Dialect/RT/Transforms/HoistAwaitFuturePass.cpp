// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Analysis/StaticLoops.h>
#include <concretelang/Dialect/RT/IR/RTDialect.h>
#include <concretelang/Dialect/RT/IR/RTOps.h>
#include <concretelang/Dialect/RT/Transforms/Passes.h>

#include <iterator>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <optional>

namespace {
struct HoistAwaitFuturePass
    : public HoistAwaitFuturePassBase<HoistAwaitFuturePass> {
  struct ExtractAndInsert {
    mlir::tensor::ExtractSliceOp extract;
    mlir::tensor::ParallelInsertSliceOp insert;
  };

  template <typename RangeT> static size_t getRangeLength(RangeT &&r) {
    return std::distance(r.begin(), r.end());
  }

  bool allStatic(llvm::ArrayRef<mlir::OpFoldResult> a) {
    return llvm::all_of(
        a, [](mlir::OpFoldResult r) { return r.is<mlir::Attribute>(); });
  }

  static std::optional<ExtractAndInsert> getExtractAndInsertOps(mlir::Value v) {
    if (getRangeLength(v.getUses()) == 2) {
      auto it1 = v.getUses().begin();
      auto it0 = it1++;

      std::optional<ExtractAndInsert> ret;

      auto castpair =
          [](mlir::Operation *op0,
             mlir::Operation *op1) -> std::optional<ExtractAndInsert> {
        mlir::tensor::ExtractSliceOp extract =
            llvm::dyn_cast<mlir::tensor::ExtractSliceOp>(op0);
        mlir::tensor::ParallelInsertSliceOp insert =
            llvm::dyn_cast<mlir::tensor::ParallelInsertSliceOp>(op1);

        return (extract && insert)
                   ? std::make_optional(ExtractAndInsert{extract, insert})
                   : std::nullopt;
      };

      if ((ret = castpair(it0->getOwner(), it1->getOwner())))
        return ret;
      else
        return castpair(it1->getOwner(), it0->getOwner());
    }

    return std::nullopt;
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    func.walk([&](mlir::concretelang::RT::AwaitFutureOp awaitFutureOp) {
      if (!awaitFutureOp.getResult().hasOneUse())
        return;

      mlir::tensor::ParallelInsertSliceOp consumer =
          llvm::dyn_cast<mlir::tensor::ParallelInsertSliceOp>(
              awaitFutureOp.getResult().getUses().begin()->getOwner());

      if (!consumer)
        return;

      mlir::scf::ForallOp forallOp =
          llvm::dyn_cast<mlir::scf::ForallOp>(awaitFutureOp->getParentOp());

      if (!forallOp)
        return;

      mlir::Value dst = consumer.getDest();
      mlir::RankedTensorType dstType =
          dst.getType().dyn_cast<mlir::RankedTensorType>();

      if (!dstType)
        return;

      if (!llvm::any_of(forallOp.getRegionOutArgs(),
                        [=](mlir::Value output) { return output == dst; }))
        return;

      std::optional<ExtractAndInsert> eai = getExtractAndInsertOps(dst);

      if (!eai)
        return;

      if (!(eai->extract.getOffsets() == eai->insert.getOffsets() &&
            eai->extract.getStrides() == eai->insert.getStrides() &&
            eai->extract.getSizes() == eai->insert.getSizes()))
        return;

      llvm::dbgs() << "HIT HIT HIT2\n";

      if (!(allStatic(forallOp.getMixedLowerBound()) &&
            allStatic(forallOp.getMixedUpperBound()) &&
            allStatic(forallOp.getMixedStep())))
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

      // TODO: Normalized indexes

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

      rewriter.setInsertionPointAfter(forallOp);
      mlir::scf::ForallOp newForallOp = rewriter.create<mlir::scf::ForallOp>(
          forallOp.getLoc(), forallOp.getMixedLowerBound(),
          forallOp.getMixedUpperBound(), forallOp.getMixedStep(), newOutputs,
          std::nullopt);

      // Move operations from old for op to new one
      auto &newOperations = newForallOp.getBody()->getOperations();
      mlir::Block *oldBody = forallOp.getBody();

      newOperations.splice(newOperations.begin(), oldBody->getOperations(),
                           oldBody->begin(), std::prev(oldBody->end()));

      rewriter.setInsertionPointAfter(awaitFutureOp);
      mlir::Value futureAsTensor =
          rewriter.create<mlir::tensor::FromElementsOp>(
              awaitFutureOp.getLoc(),
              mlir::ValueRange{awaitFutureOp.getInput()});

      mlir::scf::InParallelOp oldTerminator = forallOp.getTerminator();
      mlir::scf::InParallelOp newTerminator = newForallOp.getTerminator();

      mlir::Block::OpListType &oldTerminatorOps =
          oldTerminator.getRegion().getBlocks().begin()->getOperations();
      mlir::Block::OpListType &newTerminatorOps =
          newTerminator.getRegion().getBlocks().begin()->getOperations();

      newTerminatorOps.splice(newTerminatorOps.begin(), oldTerminatorOps,
                              oldTerminatorOps.begin(), oldTerminatorOps.end());

      llvm::SmallVector<mlir::OpFoldResult> ones(tripCounts.size(),
                                                 rewriter.getI64IntegerAttr(1));

      mlir::Value tensorOfFuturesRegionOutArg =
          newForallOp.getRegionOutArgs().back();

      rewriter.setInsertionPointAfter(eai->insert);
      rewriter.create<mlir::tensor::ParallelInsertSliceOp>(
          eai->insert.getLoc(), futureAsTensor, tensorOfFuturesRegionOutArg,
          mlir::getAsOpFoldResult(newForallOp.getInductionVars()), ones, ones);

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

      rewriter.setInsertionPointAfter(newForallOp);
      mlir::scf::ForallOp syncForallOp = rewriter.create<mlir::scf::ForallOp>(
          forallOp.getLoc(), forallOp.getMixedLowerBound(),
          forallOp.getMixedUpperBound(), forallOp.getMixedStep(),
          mlir::ValueRange{tensorOfValues}, std::nullopt);

      mlir::Value resultTensorOfFutures = newForallOp.getResults().back();

      // TODO: Make sure that the between the await future and the parallel
      // insertion there is nothing
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
        llvm::dbgs() << "Mapping IV " << oldIV.getImpl() << " to "
                     << newIV.getImpl() << "\n";
      }

      syncMapping.map(dst, syncForallOp.getOutputBlockArguments().back());
      syncMapping.map(eai->insert.getSource(), newAwaitFutureOp.getResult());

      mlir::scf::InParallelOp syncTerminator = syncForallOp.getTerminator();
      rewriter.setInsertionPointToStart(syncTerminator.getBody());
      rewriter.clone(*eai->insert.getOperation(), syncMapping);

      llvm::SmallVector<mlir::Value> newResults;

      {
        size_t offs = 0;
        for (size_t i = 0; i < forallOp.getNumResults(); i++) {
          if (i == oldResultIdx) {
            // forallOp.getResult(i).replaceAllUsesWith(syncForallOp.getResult(0));
            newResults.push_back(syncForallOp.getResult(0));
            offs = 1;
            llvm::dbgs() << "Yes, this one!!!!\n";
          } else {
            newResults.push_back(newForallOp.getResult(i - offs));
            // forallOp.getResult(i).replaceAllUsesWith(
            //     newForallOp.getResult(i - offs));
          }
        }
      }

      // llvm::dbgs() << "New Results: " << newResults.size() << ", "
      //              << newResults[0] << "\n";

      dst.replaceAllUsesWith(
          forallOp.getOutputs().drop_front(oldResultIdx).front());
      eai->insert->erase();
      awaitFutureOp.erase();
      // oldTerminator.erase();
      // newResults[0].dump();
      // func.dump();
      rewriter.replaceOp(forallOp, newResults);
    });
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
