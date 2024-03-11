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

      for (unsigned end = loops.size(); end > 0; --end) {
        unsigned start = 0;
        for (; start < end - 1; ++start) {
          auto maxPos = *std::max_element(
              std::next(coalesceableLoopRanges.begin(), start),
              std::next(coalesceableLoopRanges.begin(), end));
          if (maxPos > start)
            continue;

          auto band = llvm::MutableArrayRef(loops.data() + start, end - start);
          (void)mlir::coalesceLoops(band);
          break;
        }
        // If a band was found and transformed, keep looking at the loops above
        // the outermost transformed loop.
        if (start != end - 1)
          end = start + 1;
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::concretelang::createCollapseParallelLoops() {
  return std::make_unique<CollapseParallelLoopsPass>();
}
