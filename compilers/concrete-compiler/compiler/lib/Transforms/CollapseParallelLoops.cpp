// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace {
struct LoopParams {
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
};

/// Return the new lower bound, upper bound, and step in that order. Insert any
/// additional bounds calculations before the given builder and any additional
/// conversion back to the original loop induction value inside the given Block.
static LoopParams normalizeLoop(mlir::OpBuilder &boundsBuilder,
                                mlir::OpBuilder &insideLoopBuilder,
                                mlir::Location loc, mlir::Value lowerBound,
                                mlir::Value upperBound, mlir::Value step,
                                mlir::Value inductionVar) {
  // Check if the loop is already known to have a constant zero lower bound or
  // a constant one step.
  bool isZeroBased = false;
  if (auto ubCst = lowerBound.getDefiningOp<mlir::arith::ConstantIndexOp>())
    isZeroBased = ubCst.value() == 0;

  bool isStepOne = false;
  if (auto stepCst = step.getDefiningOp<mlir::arith::ConstantIndexOp>())
    isStepOne = stepCst.value() == 1;

  // Compute the number of iterations the loop executes: ceildiv(ub - lb, step)
  // assuming the step is strictly positive.  Update the bounds and the step
  // of the loop to go from 0 to the number of iterations, if necessary.
  if (isZeroBased && isStepOne)
    return {/*lowerBound=*/lowerBound, /*upperBound=*/upperBound,
            /*step=*/step};

  mlir::Value diff =
      boundsBuilder.create<mlir::arith::SubIOp>(loc, upperBound, lowerBound);
  mlir::Value newUpperBound =
      boundsBuilder.create<mlir::arith::CeilDivSIOp>(loc, diff, step);

  mlir::Value newLowerBound =
      isZeroBased ? lowerBound
                  : boundsBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value newStep =
      isStepOne ? step
                : boundsBuilder.create<mlir::arith::ConstantIndexOp>(loc, 1);

  // Insert code computing the value of the original loop induction variable
  // from the "normalized" one.
  mlir::Value scaled = isStepOne
                           ? inductionVar
                           : insideLoopBuilder.create<mlir::arith::MulIOp>(
                                 loc, inductionVar, step);
  mlir::Value shifted =
      isZeroBased ? scaled
                  : insideLoopBuilder.create<mlir::arith::AddIOp>(loc, scaled,
                                                                  lowerBound);

  mlir::SmallPtrSet<mlir::Operation *, 2> preserve{scaled.getDefiningOp(),
                                                   shifted.getDefiningOp()};
  inductionVar.replaceAllUsesExcept(shifted, preserve);
  return {/*lowerBound=*/newLowerBound, /*upperBound=*/newUpperBound,
          /*step=*/newStep};
}

/// Transform a loop with a strictly positive step
///   for %i = %lb to %ub step %s
/// into a 0-based loop with step 1
///   for %ii = 0 to ceildiv(%ub - %lb, %s) step 1 {
///     %i = %ii * %s + %lb
/// Insert the induction variable remapping in the body of `inner`, which is
/// expected to be either `loop` or another loop perfectly nested under `loop`.
/// Insert the definition of new bounds immediate before `outer`, which is
/// expected to be either `loop` or its parent in the loop nest.
static void normalizeLoop(mlir::scf::ForOp loop, mlir::scf::ForOp outer,
                          mlir::scf::ForOp inner) {
  mlir::OpBuilder builder(outer);
  mlir::OpBuilder innerBuilder = mlir::OpBuilder::atBlockBegin(inner.getBody());
  auto loopPieces = normalizeLoop(builder, innerBuilder, loop.getLoc(),
                                  loop.getLowerBound(), loop.getUpperBound(),
                                  loop.getStep(), loop.getInductionVar());

  loop.setLowerBound(loopPieces.lowerBound);
  loop.setUpperBound(loopPieces.upperBound);
  loop.setStep(loopPieces.step);
}

static mlir::LogicalResult
coalesceLoops_(llvm::MutableArrayRef<mlir::scf::ForOp> loops) {
  if (loops.size() < 2)
    return mlir::failure();

  mlir::scf::ForOp innermost = loops.back();
  mlir::scf::ForOp outermost = loops.front();

  // 1. Make sure all loops iterate from 0 to upperBound with step 1.  This
  // allows the following code to assume upperBound is the number of iterations.
  for (auto loop : loops)
    normalizeLoop(loop, outermost, innermost);

  // 2. Emit code computing the upper bound of the coalesced loop as product
  // of the number of iterations of all loops.
  mlir::OpBuilder builder(outermost);
  mlir::Location loc = outermost.getLoc();
  mlir::Value upperBound = outermost.getUpperBound();
  for (auto loop : loops.drop_front())
    upperBound = builder.create<mlir::arith::MulIOp>(loc, upperBound,
                                                     loop.getUpperBound());
  outermost.setUpperBound(upperBound);

  for (unsigned i = loops.size() - 1; i > 0; --i) {
    // replaceIterArgsAndYieldResults(innermost);
    auto iterOperands = loops[i].getIterOperands();
    auto iterArgs = loops[i].getRegionIterArgs();
    for (auto e : llvm::zip(iterOperands, iterArgs))
      std::get<1>(e).replaceAllUsesWith(std::get<0>(e));
  }
  builder.setInsertionPointToStart(outermost.getBody());

  // 3. Remap induction variables. For each original loop, the value of the
  // induction variable can be obtained by dividing the induction variable of
  // the linearized loop by the total number of iterations of the loops nested
  // in it modulo the number of iterations in this loop (remove the values
  // related to the outer loops):
  //   iv_i = floordiv(iv_linear, product-of-loop-ranges-until-i) mod range_i.
  // Compute these iteratively from the innermost loop by creating a "running
  // quotient" of division by the range.
  mlir::Value previous = outermost.getInductionVar();
  for (unsigned i = 0, e = loops.size(); i < e; ++i) {
    unsigned idx = loops.size() - i - 1;
    if (i != 0)
      previous = builder.create<mlir::arith::DivSIOp>(
          loc, previous, loops[idx + 1].getUpperBound());

    mlir::Value iv = (i == e - 1)
                         ? previous
                         : builder.create<mlir::arith::RemSIOp>(
                               loc, previous, loops[idx].getUpperBound());
    replaceAllUsesInRegionWith(loops[idx].getInductionVar(), iv,
                               loops.back().getRegion());
  }

  //   llvm::errs() << "Coalesce : \n"
  // 	       << outermost << "\n"
  // 	       << loops[1] << "\n"
  // 	       << innermost << "\n"
  // 	       << innermost.getBody()->back() << "\n"
  // 	       << outermost.getBody()->back() << "\n\n";

  // 4. Move the operations from the innermost just above the second-outermost
  // loop, delete the extra terminator and the second-outermost loop.
  mlir::scf::ForOp second = loops[1];
  // innermost.getBody()->back().erase();
  outermost.getBody()->getOperations().splice(
      mlir::Block::iterator(second.getOperation()),
      innermost.getBody()->getOperations());
  outermost.getBody()->back().erase();
  // for (unsigned i = loops.size() - 1; i > 0; --i) {
  // llvm::errs() << "Deleting : " << loops[i] << "\n";
  // for (auto &op : loops[i].getBody()->getOperations())
  //   op.erase();
  // }
  // loops[i].getBody()->back().erase();
  // if (!second->use_empty())
  // llvm::errs() << "Leftoverss: "  << "\n";
  // second->dropAllUses();
  second.erase();
  // llvm::errs() << "Second : \n"
  //<< second << "\n";
  llvm::errs() << "Coalesced : \n" << outermost << "\n";
  return mlir::success();
}

struct CollapseParallelLoopsPass
    : public CollapseParallelLoopsBase<CollapseParallelLoopsPass> {

  /// Walk either an scf.for or an affine.for to find a band to coalesce.
  template <typename LoopOpTy> static void walkLoop(LoopOpTy op) {}

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    module.walk([&](mlir::scf::ForOp forOp) {
      // Ignore nested loops.
      if (forOp->getParentOfType<mlir::scf::ForOp>())
        return;

      // Determine which sequences of nested loops can be coalesced
      // TODO: add loop interchange and hoisting to find more
      // opportunities by getting multiple parallel loops in sequence
      mlir::SmallVector<mlir::scf::ForOp, 4> loops;
      getPerfectlyNestedLoops(loops, forOp);
      mlir::SmallVector<unsigned, 4> coalesceableLoopRanges(loops.size());
      for (unsigned i = 0, e = loops.size(); i < e; ++i) {
        // Any loop is coalesceable to itself
        coalesceableLoopRanges[i] = i;

        // The outermost loop doesn't have any outer loop to collapse into
        if (i == 0)
          continue;

        // A loop will only be coalesced with another if both are
        // parallel.  Otherwise it is irrelevant in this pass.
        // If this loop itself is not parallel, then nothing we can do.
        auto attr = loops[i]->getAttrOfType<mlir::BoolAttr>("parallel");
        if (attr == nullptr || attr.getValue() == false)
          continue;

        auto areValuesDefinedAbove = [&](mlir::ValueRange values,
                                         mlir::Region &limit) -> bool {
          for (mlir::Value v : values)
            if (!v.getParentRegion()->isProperAncestor(&limit) &&
                !v.isa<mlir::BlockArgument>())
              return false;
          return true;
        };

        // Find how many loops are able to be coalesced
        for (unsigned j = 0; j < i; ++j) {
          if (areValuesDefinedAbove(loops[i].getOperands(),
                                    loops[j].getRegion())) {
            coalesceableLoopRanges[i] = j;
            break;
          }
        }
        //  Now ensure that all loops in this sequence
        //  [coalesceableLoopRanges[i], i] are parallel. Otherwise
        //  update the range's lower bound.
        for (int k = i - 1; k >= (int)coalesceableLoopRanges[i]; --k) {
          auto attrK = loops[k]->getAttrOfType<mlir::BoolAttr>("parallel");
          if (attrK == nullptr || attrK.getValue() == false) {
            coalesceableLoopRanges[i] = k + 1;
            break;
          }
        }
      }

      //       llvm::errs() << "Nesting candidate: " << forOp << "\n";
      //       for (unsigned i = 0; i < coalesceableLoopRanges.size(); ++i)
      // 	llvm::errs() << "coalesceableLoopRanges[" << i << "] : " <<
      // coalesceableLoopRanges[i] << "\n";

      for (unsigned end = loops.size(); end > 0; --end) {
        unsigned start = 0;
        for (; start < end - 1; ++start) {
          auto maxPos = *std::max_element(
              std::next(coalesceableLoopRanges.begin(), start),
              std::next(coalesceableLoopRanges.begin(), end));
          // llvm::errs() << "Max pos: " << maxPos << "  start : " << start <<
          // "\n";
          if (maxPos > start)
            continue;
          // llvm::errs() << "Coalescing: " << "\n";

          auto band = llvm::MutableArrayRef(loops.data() + start, end - start);
          (void)coalesceLoops_(band);
          break;
        }
        // If a band was found and transformed, keep looking at the loops above
        // the outermost transformed loop.
        if (start != end - 1)
          end = start + 1;
      }
      // llvm::errs() << "POST: " << forOp << "\n";
    });
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::concretelang::createCollapseParallelLoops() {
  return std::make_unique<CollapseParallelLoopsPass>();
}
