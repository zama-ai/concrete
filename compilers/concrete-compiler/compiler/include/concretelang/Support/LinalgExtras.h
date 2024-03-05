// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_LINALG_EXTRAS_H_
#define CONCRETELANG_SUPPORT_LINALG_EXTRAS_H_

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace mlir {
namespace concretelang {
namespace linalgextras {
using namespace mlir;
using namespace mlir::linalg;

static SmallVector<Value> makeCanonicalAffineApplies(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ArrayRef<Value> vals) {
  if (map.isEmpty())
    return {};

  assert(map.getNumInputs() == vals.size());
  SmallVector<Value> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = AffineMap::get(dims, map.getNumSymbols(), e);
    SmallVector<Value> operands(vals.begin(), vals.end());
    canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(b.create<AffineApplyOp>(loc, exprMap, operands));
  }
  return res;
}

template <typename LoadOpTy, typename StoreOpTy, typename OpType>
static llvm::SmallVector<Value> inlineRegionAndEmitStore(
    OpBuilder &b, Location loc, OpType op, ArrayRef<Value> indexedValues,
    ArrayRef<SmallVector<Value>> indexing, ArrayRef<Value> outputBuffers) {
  auto &block = op->getRegion(0).front();
  IRMapping map;
  map.map(block.getArguments(), indexedValues);
  for (auto &op : block.without_terminator()) {
    auto *newOp = b.clone(op, map);
    map.map(op.getResults(), newOp->getResults());
  }

  Operation *terminator = block.getTerminator();
  llvm::SmallVector<Value> retVals;

  for (OpOperand &operand : terminator->getOpOperands()) {
    Value toStore = map.lookupOrDefault(operand.get());
    Value newTens = b.create<StoreOpTy>(
        loc, toStore, outputBuffers[operand.getOperandNumber()],
        indexing[operand.getOperandNumber()]);
    retVals.push_back(newTens);
  }

  return retVals;
}
/// Replace the index operations in the body of the loop nest by the matching
/// induction variables.
static void replaceIndexOpsByInductionVariables(LinalgOp linalgOp,
                                                PatternRewriter &rewriter,
                                                ArrayRef<Operation *> loopOps) {
  // Extract the induction variables of the loop nest from outer to inner.
  SmallVector<Value> allIvs;
  for (Operation *loopOp : loopOps) {
    llvm::TypeSwitch<Operation *>(loopOp)
        .Case([&](scf::ParallelOp parallelOp) {
          allIvs.append(parallelOp.getInductionVars().begin(),
                        parallelOp.getInductionVars().end());
        })
        .Case([&](scf::ForOp forOp) {
          allIvs.push_back(forOp.getInductionVar());
        })
        .Case([&](AffineForOp affineForOp) {
          allIvs.push_back(affineForOp.getInductionVar());
        })
        .Default([&](Operation *op) { assert(false && "unexpected op"); });
  }
  assert(linalgOp.getNumLoops() == allIvs.size() &&
         "expected the number of loops and induction variables to match");
  // Replace the index operations in the body of the innermost loop op.
  if (!loopOps.empty()) {
    LoopLikeOpInterface loopOp = loopOps.back();
    for (IndexOp indexOp :
         llvm::make_early_inc_range(loopOp.getLoopBody().getOps<IndexOp>()))
      rewriter.replaceOp(indexOp, allIvs[indexOp.getDim()]);
  }
}

template <typename LoadOpTy, typename StoreOpTy>
static llvm::SmallVector<Value>
emitScalarImplementation(OpBuilder &b, Location loc, ArrayRef<Value> allIvs,
                         LinalgOp linalgOp, ValueRange operandValuesToUse) {
  assert(linalgOp.hasTensorSemantics() &&
         "expected linalg op with buffer semantics");
  SmallVector<Value> indexedValues;
  indexedValues.reserve(linalgOp->getNumOperands());

  auto allIvsPlusDims = SmallVector<Value>(allIvs.begin(), allIvs.end());

  // TODO: Avoid the loads if the corresponding argument of the
  // region has no uses.
  // 1.a. Emit load from input operand or for scalars access the operand itself.
  for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
    Value v = operandValuesToUse[inputOperand->getOperandNumber()];

    if (linalgOp.isScalar(inputOperand)) {
      indexedValues.push_back(v);
      continue;
    }
    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(inputOperand), allIvsPlusDims);
    indexedValues.push_back(b.create<LoadOpTy>(loc, v, indexing));
  }
  // 1.b. Emit load from output views.
  for (OpOperand *outputOperand : linalgOp.getDpsInitOperands()) {
    Value v = operandValuesToUse[outputOperand->getOperandNumber()];

    SmallVector<Value> indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(outputOperand), allIvsPlusDims);
    indexedValues.push_back(b.create<LoadOpTy>(loc, v, indexing));
  }

  // TODO: When a region inliner exists, use it.
  // 2. Inline region, currently only works for a single basic block.
  // 3. Emit store.
  SmallVector<SmallVector<Value>, 8> indexing;
  SmallVector<Value> outputBuffers;
  for (OpOperand *outputOperand : linalgOp.getDpsInitOperands()) {
    if (outputOperand->get().getType().isa<mlir::TensorType>()) {
      indexing.push_back(makeCanonicalAffineApplies(
          b, loc, linalgOp.getMatchingIndexingMap(outputOperand),
          allIvsPlusDims));
      outputBuffers.push_back(operandValuesToUse.back());
    }
  }
  return inlineRegionAndEmitStore<LoadOpTy, StoreOpTy>(
      b, loc, linalgOp, indexedValues, indexing, outputBuffers);
}

template <typename LoopTy>
static FailureOr<LinalgLoops>
linalgTensorOpToLoopsImpl(PatternRewriter &rewriter, LinalgOp linalgOp,
                          bool parallelizeLoops) {
  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (which is asserted in the inverse calculation).
  assert(linalgOp.hasTensorSemantics() &&
         "expected linalg op with value semantics");

  auto loopRanges = linalgOp.createLoopRanges(rewriter, linalgOp.getLoc());
  auto iteratorTypes = llvm::to_vector<4>(linalgOp.getIteratorTypesArray());

  SmallVector<Value> allIvs;
  GenerateLoopNest<LoopTy>::doit(
      rewriter, linalgOp.getLoc(), loopRanges, linalgOp, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
        // assert(operandValuesToUse == linalgOp->getOperands() &&
        //        "expect operands are captured and not passed by loop
        //        argument");
        allIvs.append(ivs.begin(), ivs.end());
        return emitScalarImplementation<tensor::ExtractOp, tensor::InsertOp>(
            b, loc, allIvs, linalgOp, operandValuesToUse);
        //        return scf::ValueVector{};
      });
  // Number of loop ops might be different from the number of ivs since some
  // loops like affine.parallel and scf.parallel have multiple ivs.
  SetVector<Operation *> loopSet;
  for (Value iv : allIvs) {
    if (!iv)
      return failure();
    // The induction variable is a block argument of the entry block of the
    // loop operation.
    BlockArgument ivVal = iv.dyn_cast<BlockArgument>();
    if (!ivVal)
      return failure();
    loopSet.insert(ivVal.getOwner()->getParentOp());
  }
  LinalgLoops loops(loopSet.begin(), loopSet.end());
  // Just mark loop with a parallel attributes
  if (parallelizeLoops) {
    for (auto loop : llvm::enumerate(loops)) {
      loop.value()->setAttr("parallel", rewriter.getBoolAttr(isParallelIterator(
                                            iteratorTypes[loop.index()])));
    }
  }
  // Replace all index operations in the loop body.
  replaceIndexOpsByInductionVariables(linalgOp, rewriter, loops);
  return loops;
}
} // namespace linalgextras
} // namespace concretelang
} // namespace mlir

#endif
