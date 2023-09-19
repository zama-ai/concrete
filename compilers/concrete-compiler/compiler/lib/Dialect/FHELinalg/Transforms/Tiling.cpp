// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "llvm/ADT/SmallVector.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <concretelang/Dialect/FHE/IR/FHEOps.h>
#include <concretelang/Dialect/FHE/IR/FHETypes.h>
#include <concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h>
#include <concretelang/Dialect/FHELinalg/Transforms/Tiling.h>
#include <concretelang/Support/Constants.h>

namespace mlir {
namespace concretelang {

/// Marker to avoid infinite recursion of the rewriting pattern
static const mlir::StringLiteral kTransformMarker =
    "__internal_tiling_marker__";

class GenericTilingPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
public:
  GenericTilingPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::linalg::GenericOp>(
            context, ::mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  // Copied from llvm-project/mlir/lib/Dialect/Linalg/Transforms/Tiling.cpp
  static llvm::SmallVector<mlir::OpFoldResult> calculateNumThreadsFromTileSizes(
      mlir::RewriterBase &b, mlir::TilingInterface op,
      llvm::ArrayRef<mlir::OpFoldResult> tileSizes) {
    llvm::SmallVector<mlir::Range> loopRanges = op.getIterationDomain(b);
    unsigned nLoops = loopRanges.size();
    llvm::SmallVector<mlir::OpFoldResult> numThreads;
    numThreads.reserve(nLoops);
    mlir::AffineExpr s0, s1;
    mlir::bindSymbols(b.getContext(), s0, s1);
    mlir::AffineExpr divExpr = s0.ceilDiv(s1);

    for (const auto &it : llvm::zip(tileSizes, loopRanges)) {
      mlir::OpFoldResult numTiles = std::get<0>(it);
      if (!mlir::isConstantIntValue(numTiles, 0))
        numTiles = mlir::makeComposedFoldedAffineApply(
            b, op.getLoc(), divExpr, {std::get<1>(it).size, std::get<0>(it)});
      numThreads.push_back(numTiles);
    }

    return numThreads;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->hasAttr(kTransformMarker) || !op->hasAttr("tile-sizes"))
      return mlir::failure();

    mlir::ArrayAttr tileSizesAttr =
        op->getAttrOfType<mlir::ArrayAttr>("tile-sizes");

    if (!tileSizesAttr) {
      op->emitError("Wrong type for attribute \"tile-sizes\"");
      return mlir::failure();
    }

    llvm::SmallVector<OpFoldResult> tileSizes;

    for (mlir::Attribute size : tileSizesAttr)
      tileSizes.push_back(size);

    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes =
        op.getIteratorTypesArray();

    mlir::TilingInterface tileableOp =
        llvm::dyn_cast<mlir::TilingInterface>(op.getOperation());

    assert(tileableOp);

    // If the iterator types are all parallel, just use a tiled
    // parallel loop
    if (llvm::all_of(iteratorTypes, [](mlir::utils::IteratorType itty) {
          return itty == mlir::utils::IteratorType::parallel;
        })) {
      mlir::FailureOr<mlir::linalg::ForallTilingResult> res =
          mlir::linalg::tileToForallOpUsingTileSizes(rewriter, tileableOp,
                                                     tileSizes, std::nullopt);

      mlir::LogicalResult lres = res;

      if (lres.succeeded()) {
        res.value().tileOp->setAttr(kTransformMarker, rewriter.getUnitAttr());
        res.value().tiledOp->setAttr(kTransformMarker, rewriter.getUnitAttr());
        rewriter.replaceOp(op.getOperation(), res.value().tileOp->getResults());
      }

      return res;
    }

    // If all, but the last iterator types are parallel and the last
    // type is a reduction, tile the reduction
    if (iteratorTypes.size() > 1 &&
        std::all_of(iteratorTypes.begin(), iteratorTypes.end() - 1,
                    [](mlir::utils::IteratorType itty) {
                      return itty == mlir::utils::IteratorType::parallel;
                    }) &&
        *(iteratorTypes.end() - 1) == mlir::utils::IteratorType::reduction) {

      llvm::SmallVector<mlir::OpFoldResult> numThreads =
          calculateNumThreadsFromTileSizes(rewriter, tileableOp, tileSizes);

      mlir::PartialReductionOpInterface reductionOp =
          llvm::dyn_cast<mlir::PartialReductionOpInterface>(op.getOperation());

      mlir::FailureOr<mlir::linalg::ForallReductionTilingResult> res =
          mlir::linalg::tileReductionUsingForall(
              rewriter, reductionOp, numThreads, tileSizes, std::nullopt,
              [](mlir::Operation *op,
                 mlir::OpBuilder &b) -> std::optional<mlir::Value> {
                if (llvm::isa<concretelang::FHE::AddEintOp>(op) ||
                    llvm::isa<concretelang::FHE::AddEintIntOp>(op)) {
                  return b.create<concretelang::FHE::ZeroEintOp>(
                      op->getLoc(), op->getResult(0).getType());
                }

                return std::nullopt;
              });

      mlir::LogicalResult lres = res;

      if (lres.succeeded()) {
        res.value().parallelTiledOp->setAttr(kTransformMarker,
                                             rewriter.getUnitAttr());
        res.value().mergeOp->setAttr(kTransformMarker, rewriter.getUnitAttr());
        res.value().initialOp->setAttr(kTransformMarker,
                                       rewriter.getUnitAttr());
      }

      return res;
    }

    return mlir::failure();
  }
};

/// Perfoms the actual tiling of `FHELinalg.matmul_eint_int`
/// operations that have been marked with a "tile-sizes" attribute.
class LinalgTilingPass : public LinalgTilingBase<LinalgTilingPass> {
public:
  void runOnOperation() override {
    mlir::Operation *op = getOperation();

    mlir::RewritePatternSet patterns(op->getContext());
    patterns.add<GenericTilingPattern>(op->getContext());

    if (mlir::applyPatternsAndFoldGreedily(op, std::move(patterns)).failed()) {
      this->signalPassFailure();
    }

    op->walk([](mlir::Operation *op) { op->removeAttr(kTransformMarker); });
  }
};

/// Marks all `FHELinalg.matmul_eint_int` operations that with a
/// "tile-sizes" attribute containing the specified tile sizes.
class FHELinalgTilingMarkerPass
    : public FHELinalgTilingMarkerBase<FHELinalgTilingMarkerPass> {
public:
  FHELinalgTilingMarkerPass(llvm::ArrayRef<int64_t> tileSizes)
      : tileSizes(tileSizes.vec()) {}

  void runOnOperation() override {
    mlir::Operation *op = getOperation();

    mlir::ArrayAttr tileAttr =
        mlir::Builder(&this->getContext()).getI64ArrayAttr(tileSizes);

    op->walk([&](mlir::concretelang::FHELinalg::MatMulEintIntOp matmulOp) {
      matmulOp.getOperation()->setAttr("tile-sizes", tileAttr);
    });
  }

protected:
  std::vector<int64_t> tileSizes;
};

std::unique_ptr<mlir::OperationPass<>> createLinalgTilingPass() {
  return std::make_unique<LinalgTilingPass>();
}

std::unique_ptr<mlir::OperationPass<>>
createFHELinalgTilingMarkerPass(llvm::ArrayRef<int64_t> tileSizes) {
  return std::make_unique<FHELinalgTilingMarkerPass>(tileSizes);
}
} // namespace concretelang
} // namespace mlir
