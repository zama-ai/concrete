// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Dialect/TFHE/IR/TFHEDialect.h>
#include <concretelang/Dialect/TFHE/IR/TFHEOps.h>
#include <concretelang/Dialect/TFHE/IR/TFHETypes.h>
#include <functional>
#include <limits>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/RegionUtils.h>

#include <concretelang/Analysis/StaticLoops.h>
#include <concretelang/Interfaces/BatchableInterface.h>
#include <concretelang/Transforms/Passes.h>

namespace mlir {
namespace concretelang {

template <typename IndexedOpTy> struct IndexedOpInfo {
  static mlir::SmallVector<mlir::OpFoldResult> getOffsets(IndexedOpTy op);
  static mlir::Value getTensor(IndexedOpTy op);
  static int64_t getSize(IndexedOpTy op, int64_t dim);
  static int64_t getStride(IndexedOpTy op, int64_t dim);
  static bool hasAllStaticSizesAndStrides(IndexedOpTy op);
};

template <> struct IndexedOpInfo<mlir::tensor::ExtractOp> {
  static mlir::SmallVector<mlir::OpFoldResult>
  getOffsets(mlir::tensor::ExtractOp op) {
    return op.getIndices();
  }

  static mlir::Value getTensor(mlir::tensor::ExtractOp op) {
    return op.getTensor();
  }

  static int64_t getSize(mlir::tensor::ExtractOp op, int64_t dim) { return 1; }
  static int64_t getStride(mlir::tensor::ExtractOp, int64_t dim) { return 1; }

  static bool hasAllStaticSizesAndStrides(mlir::tensor::ExtractOp op) {
    return true;
  }
};

template <> struct IndexedOpInfo<mlir::tensor::InsertOp> {
  static mlir::SmallVector<mlir::OpFoldResult>
  getOffsets(mlir::tensor::InsertOp op) {
    return op.getIndices();
  }

  static mlir::Value getTensor(mlir::tensor::InsertOp op) {
    return op.getDest();
  }

  static int64_t getSize(mlir::tensor::InsertOp op, int64_t dim) { return 1; }
  static int64_t getStride(mlir::tensor::InsertOp, int64_t dim) { return 1; }
  static bool hasAllStaticSizesAndStrides(mlir::tensor::InsertOp op) {
    return true;
  }
};

template <> struct IndexedOpInfo<mlir::tensor::ExtractSliceOp> {
  static mlir::SmallVector<mlir::OpFoldResult>
  getOffsets(mlir::tensor::ExtractSliceOp op) {
    return op.getMixedOffsets();
  }
  static mlir::Value getTensor(mlir::tensor::ExtractSliceOp op) {
    return op.getSource();
  }

  static int64_t getSize(mlir::tensor::ExtractSliceOp op, int64_t dim) {
    return op.getStaticSizes()[dim];
  }

  static int64_t getStride(mlir::tensor::ExtractSliceOp op, int64_t dim) {
    return op.getStaticStrides()[dim];
  }

  static bool hasAllStaticSizesAndStrides(mlir::tensor::ExtractSliceOp op) {
    for (size_t i = 0; i < op.getSizes().size(); i++) {
      if (op.isDynamicSize(i) || op.isDynamicStride(i))
        return false;
    }

    return true;
  }
};

template <> struct IndexedOpInfo<mlir::tensor::InsertSliceOp> {
  static mlir::SmallVector<mlir::OpFoldResult>
  getOffsets(mlir::tensor::InsertSliceOp op) {
    return op.getMixedOffsets();
  }
  static mlir::Value getTensor(mlir::tensor::InsertSliceOp op) {
    return op.getDest();
  }

  static int64_t getSize(mlir::tensor::InsertSliceOp op, int64_t dim) {
    return op.getStaticSizes()[dim];
  }

  static int64_t getStride(mlir::tensor::InsertSliceOp op, int64_t dim) {
    return op.getStaticStrides()[dim];
  }

  static bool hasAllStaticSizesAndStrides(mlir::tensor::InsertSliceOp op) {
    for (size_t i = 0; i < op.getSizes().size(); i++) {
      if (op.isDynamicSize(i) || op.isDynamicStride(i))
        return false;
    }

    return true;
  }
};

// Returns the intersection of two dense sets
template <typename T>
llvm::DenseSet<T> intersectSets(llvm::DenseSet<T> &a, llvm::DenseSet<T> &b) {
  llvm::DenseSet<T> res;

  for (T element : a) {
    if (b.contains(element))
      res.insert(element);
  }

  return res;
}

// Returns a set with all elements of `a`, which do not appear in `b`
template <typename T>
llvm::DenseSet<T> setMinus(llvm::DenseSet<T> &a, llvm::DenseSet<T> &b) {
  llvm::DenseSet<T> res;

  for (T element : a) {
    if (!b.contains(element)) {
      res.insert(element);
    }
  }

  return res;
}

// Returns a vector, which contains all elements of `a` that appear
// in `f`. Order is preserved.
template <typename T>
llvm::SmallVector<T> filterVector(llvm::SmallVector<T> &a,
                                  llvm::DenseSet<T> &f) {
  llvm::SmallVector<T> res;

  for (T &elt : a) {
    if (f.contains(elt))
      res.push_back(elt);
  }

  return res;
}

// Returns the index of the first operand of `op` that uses `v`. The
// value `v` must be referenced by at least one operand, otherwise an
// assertion is triggered.
unsigned getOperandIndexForValue(mlir::Operation *op, mlir::Value v) {
  for (auto it : llvm::enumerate(op->getOperands())) {
    if (it.value() == v)
      return it.index();
  }

  llvm_unreachable("Attempted to get operand index of value that is not an "
                   "operand of the operation");
}

// Walks up the use-def-chain of of the value `v`, executing `cb`
// for any value not previously encountered un `visited`.
static void walkUseDefChainRec(mlir::DenseSet<mlir::Value> &visited,
                               mlir::Value v,
                               llvm::function_ref<void(mlir::Value)> cb) {
  if (visited.contains(v))
    return;

  cb(v);

  if (mlir::Operation *op = v.getDefiningOp()) {
    for (mlir::Value operand : op->getOperands()) {
      walkUseDefChainRec(visited, operand, cb);
    }
  }
}

// Walks up the use-def-chain of of the value `v`, executing `cb` once
// for every value encountered.
static void walkUseDefChain(mlir::Value v,
                            llvm::function_ref<void(mlir::Value)> cb) {
  mlir::DenseSet<mlir::Value> visited;

  walkUseDefChainRec(visited, v, cb);
}

// Helper function that applies a function `func` to each element of
// a container `ins` and returns the result as a `llvm::SmallVector`.
template <typename ContainerTy, typename FuncTy>
auto map(ContainerTy &&ins, FuncTy func) {
  return llvm::to_vector(llvm::map_range(ins, func));
}

// Returns true if the operation `op` is the only direct user of `v`.
static bool isSoleUser(mlir::Value v, mlir::Operation *op) {
  return !v.getUsers().empty() &&
         std::next(v.getUsers().begin()) == v.getUsers().end() &&
         (*v.getUsers().begin()) == op;
}

// Sorts the `scf.for` loops from `forOps` from the innermost to the
// outermost loop. The loops must be embedded one into another.
template <typename ContainerTy>
SmallVector<mlir::scf::ForOp>
sortLoopsInnermostToOutermost(ContainerTy &forOps) {
  SmallVector<mlir::scf::ForOp> res;

  for (mlir::scf::ForOp forOp : forOps) {
    size_t i = 0;
    for (; i < res.size(); i++) {
      if (!forOp->isAncestor(res[i]))
        break;
    }

    res.insert(res.begin() + i, forOp);
  }

  return res;
}

// Sorts the `scf.for` loops from `forOps` from the outermost to the
// innermost loop. The loops must be embedded one into another.
template <typename ContainerTy>
SmallVector<mlir::scf::ForOp>
sortLoopsOutermostToInnermost(ContainerTy &forOps) {
  SmallVector<mlir::scf::ForOp> nest = sortLoopsInnermostToOutermost(forOps);
  std::reverse(nest.begin(), nest.end());
  return nest;
}

// Takes a set of loops `forOps` and finds the longest sequence of
// perfectly nested loops starting with innermost loops of `forOps`,
// in which the predicate `parentChildPredicate` holds for all loops
// and their direct child loops if specified.
template <typename ContainerTy>
void getLongestPerfectLoopnest(
    ContainerTy forOps, mlir::scf::ForOp &innermost,
    mlir::scf::ForOp &outermost,
    llvm::function_ref<bool(mlir::scf::ForOp, mlir::scf::ForOp)>
        parentChildPredicate = nullptr) {
  assert(forOps.size() > 0);

  SmallVector<mlir::scf::ForOp> innermostToOutermost =
      sortLoopsInnermostToOutermost(forOps);

  innermost = innermostToOutermost[0];
  outermost = innermost;

  for (size_t i = 1; i < innermostToOutermost.size(); i++) {
    if (outermost->getParentOp() != innermostToOutermost[i].getOperation())
      break;

    if (parentChildPredicate &&
        !parentChildPredicate(innermostToOutermost[i], outermost)) {
      break;
    }

    outermost = innermostToOutermost[i];
  }
}

// Returns a value that corresponds to the tensor `v` with all
// dimensions, but the last `trailingDimensions` dimensions collapsed
// into a single dimension.
static mlir::Value flattenTensor(mlir::ImplicitLocOpBuilder &builder,
                                 mlir::Value v,
                                 unsigned trailingDimensions = 0) {
  mlir::RankedTensorType type = v.getType().dyn_cast<mlir::RankedTensorType>();
  assert(type && "Value type is not a ranked tensor");

  if (type.getShape().size() - trailingDimensions == 1) {
    return v;
  } else {
    mlir::ReassociationIndices prefixCollapseGroup;
    llvm::SmallVector<mlir::ReassociationIndices> collapseGroups;

    for (unsigned i = 0; i < type.getShape().size() - trailingDimensions; i++)
      prefixCollapseGroup.push_back(i);

    collapseGroups.push_back(prefixCollapseGroup);

    for (unsigned i = type.getShape().size() - trailingDimensions;
         i < type.getShape().size(); i++) {
      mlir::ReassociationIndices suffixGroup;
      suffixGroup.push_back(i);
      collapseGroups.push_back(suffixGroup);
    }

    return builder.create<mlir::tensor::CollapseShapeOp>(v, collapseGroups);
  }
}

// Returns a tensor with all the elements of the flat tensor `v`, but
// shaped as a tensor with the type `targetType`.
static mlir::Value unflattenTensor(mlir::ImplicitLocOpBuilder &builder,
                                   mlir::Value v,
                                   mlir::RankedTensorType targetType) {
  mlir::RankedTensorType type = v.getType().dyn_cast<mlir::RankedTensorType>();
  assert(type && type.getShape().size() == 1 &&
         "Value is not a tensor of rank 1");

  if (targetType.getShape().size() == 1) {
    return v;
  } else {
    mlir::ReassociationIndices expandGroup;

    for (unsigned i = 0; i < targetType.getShape().size(); i++)
      expandGroup.push_back(i);

    return builder.create<mlir::tensor::ExpandShapeOp>(
        targetType, v, llvm::ArrayRef{expandGroup});
  }
}

// Rewrites a perfect loop nest yielding M results from the innermost
// loop, such that the n-th of these results is omitted, causing the
// nest to return M-1 results.
//
// Example:
//
//   scf.for ... {
//     %v0:5 = scf.for ... {
//       %v1:5 = scf.for ... {
//         %v2:5 = scf.for ... {
//           %inner0 = ...
//           %inner1 = ...
//           %inner2 = ...
//           %inner3 = ...
//           %inner4 = ...
//           ...
//           scf.yield %inner0, %inner1, %inner2, %inner3, %inner4
//         }
//         scf.yield %v2#0, %v2#1, %v2#2, %v2#3, %v2#4
//       }
//       scf.yield %v1#0, %v1#1, %v1#2, %v1#3, %v1#4
//     }
//     scf.yield %v0#0, %v0#1, %v0#2, %v0#3, %v0#4
//   }
//
// with n=2 becomes:
//
//   scf.for ... {
//     %v0:4 = scf.for ... {
//       %v1:4 = scf.for ... {
//         %v2:4 = scf.for ... {
//           %inner0 = ...
//           %inner1 = ...
//           %inner2 = ...
//           %inner3 = ...
//           %inner4 = ...
//           ...
//           scf.yield %inner0, %inner1, %%inner3, %inner4
//         }
//         scf.yield %v2#0, %v2#1, %v2#2, %v2#3
//       }
//       scf.yield %v1#0, %v1#1, %v1#2, %v1#3
//     }
//     scf.yield %v0#0, %v0#1, %v0#2, %v0#3
//   }
static void rewritePerfectLoopNestWithReplacedNthResult(
    mlir::PatternRewriter &rewriter, mlir::scf::ForOp innermostFor,
    mlir::scf::ForOp outermostFor, unsigned n, mlir::Value replacement) {
  // Assemble loop nest from innermost and outermost loop
  llvm::SmallVector<mlir::scf::ForOp> nest;

  for (mlir::Operation *forOp = innermostFor.getOperation();
       forOp != outermostFor.getOperation()->getParentOp();
       forOp = forOp->getParentOp()) {
    nest.push_back(llvm::dyn_cast<mlir::scf::ForOp>(forOp));
  }

  // Dismiss n-th operand from all yields from outermost to innermost loop
  for (mlir::scf::ForOp currFor : llvm::reverse(nest)) {
    // Build new, empty loop nest
    rewriter.setInsertionPoint(currFor);

    SmallVector<mlir::Value> newInitArgs;
    for (auto i : llvm::enumerate(currFor.getInitArgs())) {
      if (i.index() != n) {
        newInitArgs.push_back(i.value());
      }
    }

    scf::ForOp newFor = rewriter.create<mlir::scf::ForOp>(
        currFor.getLoc(), currFor.getLowerBound(), currFor.getUpperBound(),
        currFor.getStep(), newInitArgs,
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {});

    // Copy all attributes from old for loop
    newFor->setAttrs(currFor->getAttrs());

    // Move operations from old for op to new one without yield
    mlir::Block *newBody = newFor.getBody();
    auto &newOperations = newBody->getOperations();
    mlir::Block *oldBody = currFor.getBody();

    auto end = oldBody->end();
    end--;
    newOperations.splice(newOperations.begin(), oldBody->getOperations(),
                         oldBody->begin(), end);

    mlir::scf::YieldOp oldYield =
        llvm::dyn_cast<mlir::scf::YieldOp>(currFor.getBody()->getTerminator());

    // Assemble list of new yielded values, i.e., all old operands
    // without the n-th operand
    llvm::SmallVector<mlir::Value> newYieldOperands;
    for (size_t i = 0; i < oldYield->getNumOperands(); i++) {
      if (i != n) {
        newYieldOperands.push_back(oldYield->getOperand(i));
      }
    }

    rewriter.setInsertionPointToEnd(newBody);
    rewriter.create<mlir::scf::YieldOp>(oldYield.getLoc(), newYieldOperands);

    // Remap iter args
    for (size_t i = 0; i < currFor.getNumRegionIterArgs(); i++) {
      if (i != n) {
        size_t idx = (i < n) ? i : i - 1;
        replaceAllUsesInRegionWith(currFor.getRegionIterArg(i),
                                   newFor.getRegionIterArg(idx),
                                   newFor.getRegion());
      } else {
        replaceAllUsesInRegionWith(currFor.getRegionIterArg(i), replacement,
                                   newFor.getRegion());
      }
    }

    // Remap IV
    replaceAllUsesInRegionWith(currFor.getInductionVar(),
                               newFor.getInductionVar(), newFor.getRegion());

    // Remap results in old yield
    for (auto i : llvm::enumerate(currFor.getInitArgs())) {
      replaceAllUsesInRegionWith(oldYield.getOperand(i.index()),
                                 currFor.getRegionIterArg(i.index()),
                                 currFor.getRegion());
    }

    // Assemble list of values that the old outermost for loop is
    // replaced with (i.e., all retained yielded values and the
    // replacement value for the n-th operand)
    if (currFor == outermostFor) {
      llvm::SmallVector<mlir::Value> newResults;
      for (size_t i = 0; i < currFor->getNumResults(); i++) {
        if (i < n) {
          newResults.push_back(newFor->getResult(i));
        } else if (i == n) {
          newResults.push_back(replacement);
        } else {
          newResults.push_back(newFor->getResult(i - 1));
        }
      }

      rewriter.replaceOp(currFor, newResults);
    } else {
      // An inner loop has been rewritten -> remap uses of results of
      // the old loop to the new loop
      mlir::scf::ForOp parentFor =
          llvm::dyn_cast<mlir::scf::ForOp>(currFor->getParentOp());

      for (auto it : llvm::enumerate(currFor.getResults())) {
        if (it.index() != n) {
          mlir::Value newResult = (it.index() < n)
                                      ? newFor.getResult(it.index())
                                      : newFor.getResult(it.index() - 1);
          replaceAllUsesInRegionWith(it.value(), newResult,
                                     parentFor.getRegion());
        }
      }
    }
  }
}

/// Checks if the value `v` is defined outside of the `loop` or a pure
/// operation that can be safely replicated outside the loop (i.e., all
/// of its operands are also recursively either defined outside of the
/// loop or pure).
static bool isHoistable(mlir::Value v, mlir::scf::ForOp loop) {
  mlir::Operation *op = v.getDefiningOp();

  return loop.isDefinedOutsideOfLoop(v) ||
         (op && mlir::isPure(op) && op->getNumResults() == 1 &&
          llvm::all_of(op->getOperands(), [&](mlir::Value operand) {
            return isHoistable(operand, loop);
          }));
}

llvm::SmallVector<mlir::Value>
buildNormalizedIndexes(mlir::PatternRewriter &rewriter,
                       llvm::ArrayRef<mlir::scf::ForOp> nest) {
  assert(nest.size() > 0);

  mlir::scf::ForOp innermost = nest[nest.size() - 1];
  llvm::SmallVector<mlir::Value> res;

  rewriter.setInsertionPointToStart(innermost.getBody());

  for (mlir::scf::ForOp forOp : nest) {
    mlir::ImplicitLocOpBuilder ilob(forOp.getLoc(), rewriter);

    mlir::Value idx = normalizeInductionVar(
        ilob, forOp.getInductionVar(), forOp.getLowerBound(), forOp.getStep());

    res.push_back(idx);
  }

  return res;
}

/// Returns an `OpFoldResult` with an `IntegerAttr` value if `v` is
/// produced by a constant, otherwise an `OpFoldResult` containing `v`
/// itself.
static mlir::OpFoldResult getValueAsOpFoldResult(mlir::Value v) {
  if (mlir::arith::ConstantOp cstOp =
          dyn_cast_or_null<mlir::arith::ConstantOp>(v.getDefiningOp())) {
    return cstOp.getValue();
  }

  return v;
}

// Checks whether the `OpFoldResult` `v` is a `mlir::Value` generated
// by a `ConstantOp`. If so, an `OpFoldResult` with an attribute corresponding
// to the value of the constant is returned. Otherwise, `v` is returned
// unchanged.
static mlir::OpFoldResult opFoldConstantValueToAttribute(mlir::OpFoldResult v) {
  if (mlir::Value dynV = v.dyn_cast<mlir::Value>()) {
    if (isConstantIndexValue(dynV)) {
      return mlir::IntegerAttr::get(
          IndexType::get(dynV.getContext()),
          llvm::APInt(64, getConstantIndexValue(dynV)));
    }
  }

  return v;
}

/// Performs an arithmetic operation on `a` and `b`, where both values
/// can be any combination of `IntegerAttr` and `Value`.
template <typename ArithOp, typename ArithFunctor,
          typename IsNeutralElementFunctor>
mlir::OpFoldResult opFoldExpr(mlir::ImplicitLocOpBuilder &builder,
                              mlir::OpFoldResult a, mlir::OpFoldResult b) {
  static IsNeutralElementFunctor isNeutralElement;

  auto exprValVal = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return builder.create<ArithOp>(a, b);
  };

  auto exprAttrVal = [&](mlir::IntegerAttr attr, mlir::Value v) -> mlir::Value {
    mlir::Value cst =
        builder.create<mlir::arith::ConstantIndexOp>(attr.getInt());

    return exprValVal(cst, v);
  };

  auto exprValAttr = [&](mlir::Value v, mlir::IntegerAttr attr) -> mlir::Value {
    mlir::Value cst =
        builder.create<mlir::arith::ConstantIndexOp>(attr.getInt());

    return exprValVal(v, cst);
  };

  auto exprAttrAttr = [&](mlir::IntegerAttr a,
                          mlir::IntegerAttr b) -> mlir::IntegerAttr {
    static ArithFunctor f;
    return builder.getIndexAttr(f(a.getInt(), b.getInt()));
  };

  if (a.is<mlir::Value>()) {
    if (b.is<mlir::Value>()) {
      return exprValVal(a.get<mlir::Value>(), b.get<mlir::Value>());
    } else {
      mlir::IntegerAttr bAttr =
          b.get<mlir::Attribute>().cast<mlir::IntegerAttr>();

      if (isNeutralElement(bAttr.getValue().getSExtValue())) {
        return a;
      } else {
        return exprValAttr(a.get<mlir::Value>(), bAttr);
      }
    }
  } else {
    mlir::IntegerAttr aAttr =
        a.get<mlir::Attribute>().cast<mlir::IntegerAttr>();

    if (b.is<mlir::Value>()) {
      return exprAttrVal(aAttr, b.get<mlir::Value>());
    } else {
      mlir::IntegerAttr bAttr =
          b.get<mlir::Attribute>().cast<mlir::IntegerAttr>();

      if (isNeutralElement(bAttr.getValue().getSExtValue()))
        return a;
      else
        return exprAttrAttr(aAttr, bAttr);
    }
  }
}

/// Hoists the pure operation producing the value `v` out of
/// `outermostFor` recursively. All newly created mappings are
/// collected in `mapping`.
static mlir::Value hoistPure(mlir::PatternRewriter &rewriter,
                             mlir::scf::ForOp outermostFor,
                             mlir::IRMapping &mapping, mlir::Value v) {
  if (outermostFor.isDefinedOutsideOfLoop(v))
    return v;

  mlir::Operation *op = v.getDefiningOp();

  assert(op && mlir::isPure(op) && op->getNumResults() == 1);

  for (mlir::Value operand : op->getOperands()) {
    if (!mapping.contains(operand))
      mapping.map(operand, hoistPure(rewriter, outermostFor, mapping, operand));
  }

  rewriter.setInsertionPoint(outermostFor);

  mlir::Operation *clonedOp = rewriter.clone(*op, mapping);

  return clonedOp->getResult(0);
}

/// Hoists the pure operation producing the value `v` out of
/// `outermostFor` recursively.
static mlir::Value hoistPure(mlir::PatternRewriter &rewriter,
                             mlir::scf::ForOp outermostFor, mlir::Value v) {
  mlir::IRMapping mapping;
  return hoistPure(rewriter, outermostFor, mapping, v);
}

template <typename IndexedOpTy>
int64_t getSliceExtents(IndexedOpTy op, size_t dimIdx) {
  int64_t stride = IndexedOpInfo<IndexedOpTy>::getStride(op, dimIdx);
  int64_t size = IndexedOpInfo<IndexedOpTy>::getSize(op, dimIdx);

  assert(stride > 0 && size > 0 &&
         std::numeric_limits<int64_t>::max() / stride >= size);

  return stride * size;
}

/// Hoists a an operation embedded into a loop nest that and that is
/// indexed using quasi-affine expressions of the loops' IVs as a
/// tensor operation outside of the outermost loop (e.g.,
/// `tensor.extract` becomes `tensor.extract_slice`). This function
/// only takescare about calculating offsets, sizes and strides. The actual
/// tensor operation must be built by the callback function `tensorOpBuilder`.
template <typename IndexedOpTy>
mlir::Value hoistIndexedOp(
    mlir::PatternRewriter &rewriter, mlir::scf::ForOp outermostFor,
    mlir::Value tensorizedOperands, IndexedOpTy indexedOp,
    llvm::function_ref<mlir::Value(
        mlir::ImplicitLocOpBuilder &, mlir::Value,
        llvm::ArrayRef<mlir::OpFoldResult>, llvm::ArrayRef<mlir::OpFoldResult>,
        llvm::ArrayRef<mlir::OpFoldResult>, llvm::ArrayRef<bool>)>
        tensorOpBuilder) {
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides;
  llvm::SmallVector<bool> ivIndexedDims;

  rewriter.setInsertionPoint(outermostFor);
  mlir::ImplicitLocOpBuilder ilob(indexedOp.getLoc(), rewriter);

  for (auto it :
       llvm::enumerate(IndexedOpInfo<IndexedOpTy>::getOffsets(indexedOp))) {
    mlir::OpFoldResult idxExpr = it.value();
    size_t dimIdx = it.index();

    mlir::scf::ForOp forOp;
    bool isAffine = isQuasiAffineIVExpression(idxExpr, &forOp);

    int64_t stride = IndexedOpInfo<IndexedOpTy>::getStride(indexedOp, dimIdx);
    int64_t size = IndexedOpInfo<IndexedOpTy>::getSize(indexedOp, dimIdx);

    if (isAffine && forOp &&
        (forOp == outermostFor || outermostFor->isAncestor(forOp))) {
      std::optional<LoopsBoundsAndStep> bas =
          getBoundsOfQuasiAffineIVExpression(idxExpr, forOp);

      assert(bas.has_value());
      assert(bas->step != 0);

      int64_t sliceExtents = getSliceExtents(indexedOp, dimIdx);
      assert(sliceExtents == 1 || sliceExtents == bas->step);
      int64_t hoistedSliceSize = size * getStaticTripCount(*bas);
      int64_t hoistedStride = (sliceExtents == 1) ? bas->step : stride;

      offsets.push_back(rewriter.getIndexAttr(bas->lb));
      sizes.push_back(rewriter.getIndexAttr(hoistedSliceSize));
      strides.push_back(rewriter.getIndexAttr(hoistedStride));

      ivIndexedDims.push_back(true);
    } else if (isAffine || idxExpr.is<mlir::Attribute>() ||
               outermostFor.isDefinedOutsideOfLoop(
                   idxExpr.dyn_cast<mlir::Value>())) {
      offsets.push_back(opFoldConstantValueToAttribute(idxExpr));
      sizes.push_back(rewriter.getIndexAttr(size));
      strides.push_back(rewriter.getIndexAttr(stride));
      ivIndexedDims.push_back(false);
    } else {
      llvm_unreachable("Unknown type of index found");
    }
  }

  return tensorOpBuilder(ilob, tensorizedOperands, offsets, sizes, strides,
                         ivIndexedDims);
}

/// Hoists a tensor.extract operation embedded into a loop nest as a
/// `tensor.extract_slice` outside of the outermost loop of the nest
static mlir::Value hoistExtractOp(mlir::PatternRewriter &rewriter,
                                  mlir::scf::ForOp outermostFor,
                                  mlir::tensor::ExtractOp extractOp) {
  return hoistIndexedOp<mlir::tensor::ExtractOp>(
      rewriter, outermostFor, extractOp.getTensor(), extractOp,
      [](mlir::ImplicitLocOpBuilder &builder, mlir::Value tensorizedOperands,
         llvm::ArrayRef<mlir::OpFoldResult> offsets,
         llvm::ArrayRef<mlir::OpFoldResult> sizes,
         llvm::ArrayRef<mlir::OpFoldResult> strides,
         llvm::ArrayRef<bool> ivIndexedDims) -> mlir::Value {
        mlir::tensor::ExtractSliceOp slice =
            builder.create<mlir::tensor::ExtractSliceOp>(
                tensorizedOperands, offsets, sizes, strides);

        // The extract slice operation above preserves non-IV-indexed
        // dimensions of the original extract operation as 1-sized
        // dimensions, e.g., a `tensor.extract[cst, i, j, cst, k]`
        // results in a slice with the shape `1xMxNx1xK` (where M, N
        // and K are the maximum values for i, j and k, assuming a
        // loop step of 1).
        //
        // If there is any non-IV-indexed dimension, add a collapse
        // shape operation that collapses the 1-sized dimensions
        // originating from non-IV-indexed dimensions of the extract
        // operation into the IV-indexed dimensions. I.e., in the
        // above example, produce a slice with the shape `MxNxK`
        // rather than `1xMxNx1xK`.
        if (llvm::all_of(ivIndexedDims, [](bool v) { return v; })) {
          return slice;
        } else {
          llvm::SmallVector<mlir::ReassociationIndices> collapseGroups;
          mlir::ReassociationIndices currCollapseGroup;

          bool prefixDone = false;
          for (auto i : llvm::enumerate(ivIndexedDims)) {
            // If this is a non-IV-indexed dimension, accumulate
            // dimension in the current group of collapsed dimensions
            if (!i.value()) {
              currCollapseGroup.push_back(i.index());
            } else {
              // If there were only non-IV-indexed dimensions before,
              // add this first IV-indexed dimension to the current
              // group of collapsed dimensions and try to accumulate
              // with following, non-IV-indexed dimensions.
              if (!prefixDone) {
                currCollapseGroup.push_back(i.index());
                prefixDone = true;
              } else {
                collapseGroups.push_back(currCollapseGroup);
                currCollapseGroup = mlir::ReassociationIndices();
                currCollapseGroup.push_back(i.index());
              }
            }
          }

          // Add last collapse group for trailing series of
          // non-IV-indexed dimensions
          if (!currCollapseGroup.empty())
            collapseGroups.push_back(currCollapseGroup);

          mlir::tensor::CollapseShapeOp cso =
              builder.create<mlir::tensor::CollapseShapeOp>(slice,
                                                            collapseGroups);

          return cso;
        }
      });
}

/// Hoists a tensor.extract_slice operation embedded into a loop nest outside of
/// the outermost loop of the nest
static mlir::Value hoistExtractOp(mlir::PatternRewriter &rewriter,
                                  mlir::scf::ForOp outermostFor,
                                  mlir::tensor::ExtractSliceOp extractSliceOp) {
  return hoistIndexedOp<mlir::tensor::ExtractSliceOp>(
      rewriter, outermostFor, extractSliceOp.getSource(), extractSliceOp,
      [](mlir::ImplicitLocOpBuilder &builder, mlir::Value tensorizedOperands,
         llvm::ArrayRef<mlir::OpFoldResult> offsets,
         llvm::ArrayRef<mlir::OpFoldResult> sizes,
         llvm::ArrayRef<mlir::OpFoldResult> strides,
         llvm::ArrayRef<bool> ivIndexedDims) -> mlir::Value {
        mlir::tensor::ExtractSliceOp slice =
            builder.create<mlir::tensor::ExtractSliceOp>(
                tensorizedOperands, offsets, sizes, strides);

        return slice;
      });
}

/// Hoists a tensor.insert operation embedded into a loop nest as a
/// tensor.insert_slice outside of the outermost loop of the nest
static mlir::Value hoistInsertOp(mlir::PatternRewriter &rewriter,
                                 mlir::Value tensorizedOperands,
                                 mlir::Value targetTensor,
                                 mlir::scf::ForOp outermostFor,
                                 mlir::tensor::InsertOp insertOp) {
  return hoistIndexedOp<mlir::tensor::InsertOp>(
      rewriter, outermostFor, targetTensor, insertOp,
      [&](mlir::ImplicitLocOpBuilder &builder, mlir::Value targetTesor,
          llvm::ArrayRef<mlir::OpFoldResult> offsets,
          llvm::ArrayRef<mlir::OpFoldResult> sizes,
          llvm::ArrayRef<mlir::OpFoldResult> strides,
          llvm::ArrayRef<bool> ivIndexedDims) -> mlir::Value {
        return builder.create<mlir::tensor::InsertSliceOp>(
            tensorizedOperands, targetTesor, offsets, sizes, strides);
      });
}

/// Hoists a tensor.insert_slice operation embedded into a loop nest
/// outside of the outermost loop of the nest
static mlir::Value hoistInsertOp(mlir::PatternRewriter &rewriter,
                                 mlir::Value tensorizedOperands,
                                 mlir::Value targetTensor,
                                 mlir::scf::ForOp outermostFor,
                                 mlir::tensor::InsertSliceOp insertSliceOp) {
  return hoistIndexedOp<mlir::tensor::InsertSliceOp>(
      rewriter, outermostFor, targetTensor, insertSliceOp,
      [&](mlir::ImplicitLocOpBuilder &builder, mlir::Value targetTesor,
          llvm::ArrayRef<mlir::OpFoldResult> offsets,
          llvm::ArrayRef<mlir::OpFoldResult> sizes,
          llvm::ArrayRef<mlir::OpFoldResult> strides,
          llvm::ArrayRef<bool> ivIndexedDims) -> mlir::Value {
        return builder.create<mlir::tensor::InsertSliceOp>(
            tensorizedOperands, targetTesor, offsets, sizes, strides);
      });
}

// Recursively a set of values `next` with all producers within
// `forOp` and those that are just outside of it. All intermediate
// values and those collected in `next` are added to `visited`. The
// function `valPredicate` is evaluated for each encountered value. If
// the predicate does not hold for any value, `extendFrontierRec`
// halts and returns false.
bool extendFrontierRec(mlir::Value v, llvm::DenseSet<mlir::Value> &next,
                       llvm::DenseSet<mlir::Value> &visited,
                       mlir::scf::ForOp forOp,
                       llvm::function_ref<bool(mlir::Value)> valPredicate) {
  if (visited.contains(v))
    return true;

  if (forOp.isDefinedOutsideOfLoop(v)) {
    next.insert(v);
  } else {
    if (!valPredicate(v))
      return false;

    visited.insert(v);

    if (mlir::BlockArgument ba = llvm::dyn_cast<mlir::BlockArgument>(v)) {
      return true;
    } else {
      mlir::Operation *definingOp = v.getDefiningOp();
      assert(definingOp);

      for (mlir::Value operand : definingOp->getOperands()) {
        if (!extendFrontierRec(operand, next, visited, forOp, valPredicate))
          return false;
      }
    }
  }

  return true;
}

// See `extendFrontierRec`
std::optional<llvm::DenseSet<mlir::Value>>
extendFrontier(const llvm::DenseSet<mlir::Value> &prev,
               llvm::DenseSet<mlir::Value> &visited, mlir::scf::ForOp forOp,
               llvm::function_ref<bool(mlir::Value)> valPredicate) {
  llvm::DenseSet<mlir::Value> next;

  for (mlir::Value v : prev) {
    if (!extendFrontierRec(v, next, visited, forOp, valPredicate))
      return std::nullopt;
  }

  return next;
}

// Returns true if `v` is a region iteration argument of `forOp`.
static bool valueIsRegionIterArgOf(mlir::Value v, mlir::scf::ForOp forOp) {
  return llvm::any_of(forOp.getRegionIterArgs(),
                      [=](mlir::Value iterArg) { return iterArg == v; });
}

// Checks if `v` refers to the iteration argument of an `scf.for` loop
static std::optional<mlir::scf::ForOp> valueIsRegionIterArg(mlir::Value v) {
  if (mlir::BlockArgument ba = llvm::dyn_cast<mlir::BlockArgument>(v)) {
    if (mlir::scf::ForOp forOp =
            llvm::dyn_cast<mlir::scf::ForOp>(ba.getOwner()->getParentOp())) {
      if (llvm::any_of(
              forOp.getRegionIterArgs(),
              [=](mlir::BlockArgument otherArg) { return otherArg == ba; }))
        return forOp;
    }
  }

  return std::nullopt;
}

// Separates the operands of a batchable operation into a vector of
// batchable and a vector of non-batchable operands according to the
// specification of the `variant`-th batching variant of the
// operation.
static void
splitOperands(BatchableOpInterface batchableOp, unsigned variant,
              llvm::SmallVector<mlir::OpOperand *> &batchableOperands,
              llvm::SmallVector<mlir::OpOperand *> &nonBatchableOperands) {
  for (mlir::OpOperand &batchableOperand :
       batchableOp.getBatchableOperands(variant)) {
    batchableOperands.push_back(&batchableOperand);
  }

  for (mlir::OpOperand &operand : batchableOp->getOpOperands()) {
    if (llvm::none_of(batchableOp.getBatchableOperands(variant),
                      [&](mlir::OpOperand &batchableOperand) {
                        return operand.getOperandNumber() ==
                               batchableOperand.getOperandNumber();
                      })) {
      nonBatchableOperands.push_back(&operand);
    }
  }
}

// Converts the ordered sequence `operands` into an unordered set of
// operands
llvm::DenseSet<mlir::Value>
operandsToValueSet(llvm::ArrayRef<mlir::OpOperand *> operands) {
  llvm::DenseSet<mlir::Value> set;

  for (mlir::OpOperand *operand : operands)
    set.insert(operand->get());

  return set;
}

/// Pattern that replaces a batchable operation embedded into a static
/// loop nest with the batched version of the operation, e.g.,
///
///   scf.for %i = c0 to %cN step %c1 {
///     scf.for %j = c0 to %cM step %c1 {
///       scf.for %k = c0 to %cK step %c1 {
///         %s0 = tensor.extract %T[%i, %j, %k]
///         %s1 = tensor.extract %U[%k, %j, %i]
///         %res = batchable_op %s0, %s1, ...
///         ...
///       }
///     }
///   }
///
/// is replaced with:
///
///   %batchedSlice0 = tensor.extract_slice
///        %T[%c0, %c0, %c0] [%cN, %cM, %cK] [%c1, %c1, %c1]
///   %batchedSlice1 = tensor.extract_slice
///        %U[%c0, %c0, %c0] [%cK, %cM, %cN] [%c1, %c1, %c1]
///   %flatSlice0 = tensor.collapse_shape %batchedSlice0
///   %flatSlice1 = tensor.collapse_shape %batchedSlice1
///   %resTFlat = batchedOp %flatSlice0, %flatSlice1, ...
///   %resT = tensor.expand_shape %resTFlat
///
///   scf.for %i = c0 to %cN step %c1 {
///     scf.for %j = c0 to %cM step %c1 {
///       scf.for %k = c0 to %cK step %c1 {
///         %res = tensor.extract %resT[%i, %j, %k]
///         ...
///       }
///     }
///   }
///
/// Any index may be a quasi-affine expression on a single loop
/// induction variable, but the distance between the result for any
/// two successive values of the IV must be constant.
///
/// The element type of batched operands can either be scalar or
/// tensors themselves.
class BatchingPattern : public mlir::OpRewritePattern<mlir::func::FuncOp> {
public:
  BatchingPattern(mlir::MLIRContext *context,
                  int64_t maxBatchSize = std::numeric_limits<int64_t>::max())
      : mlir::OpRewritePattern<mlir::func::FuncOp>(context),
        maxBatchSize(maxBatchSize) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp func,
                  mlir::PatternRewriter &rewriter) const override {
    // Operation that will be hoisted out of the loop nest and
    // replaced by the batched version of the operation
    BatchableOpInterface targetOp;

    // The perfect loop nest containing the target operation from
    // the outermost to the innermost loop
    llvm::SmallVector<mlir::scf::ForOp> nest;

    // Total number of elements that within a batch
    int64_t batchSize;

    // Selected batching variant of the batchable operation
    unsigned variant;

    // Sequences of batchable and non-batchable operands of the
    // batchable operation
    llvm::SmallVector<mlir::OpOperand *> batchableOperands;
    llvm::SmallVector<mlir::OpOperand *> nonBatchableOperands;

    // Sets of values that recursively produce the batched and
    // non-batched operands and that are contained in the loop nest
    llvm::DenseSet<mlir::Value> visitedBatched;
    llvm::DenseSet<mlir::Value> visitedNonBatched;

    // Find a batchable op which is embedded into a loop nest
    func.walk([&](BatchableOpInterface scalarOp) {
      // Predicate checking whether an scf.for op is a valid candidate
      // to expand the loop nest upwards towards the outermost loop
      auto isCandidateLoop = [](mlir::scf::ForOp forOp) -> bool {
        std::function<bool(mlir::scf::ForOp forOp)> hasKSorBS =
            [&](mlir::scf::ForOp forOp) -> bool {
          for (mlir::Operation &op : forOp.getBody()->getOperations()) {
            if (llvm::isa<TFHE::KeySwitchGLWEOp, TFHE::BootstrapGLWEOp>(op))
              return true;
            if (auto nested = llvm::dyn_cast_or_null<mlir::scf::ForOp>(op);
                nested)
              if (hasKSorBS(nested))
                return true;
          }
          return false;
        };
        return hasKSorBS(forOp) && isStaticLoop(forOp);
      };

      // Only batchable operations within at least one loop are of
      // interest
      mlir::scf::ForOp innermostFor =
          llvm::dyn_cast_or_null<mlir::scf::ForOp>(scalarOp->getParentOp());

      if (!innermostFor || !isCandidateLoop(innermostFor))
        return mlir::WalkResult::skip();

      unsigned candidateVariant = 0;

      // Try all batching variants of the batchable operation in
      // sequence until all conditions for batching are met.
      for (candidateVariant = 0;
           candidateVariant < scalarOp.getNumBatchingVariants();
           candidateVariant++) {
        llvm::SmallVector<mlir::OpOperand *> candidateBatchableOperands;
        llvm::SmallVector<mlir::OpOperand *> candidateNonBatchableOperands;

        splitOperands(scalarOp, candidateVariant, candidateBatchableOperands,
                      candidateNonBatchableOperands);

        // Construct initial frontiers from the values used directly
        // by the batchable operation
        llvm::DenseSet<mlir::Value> frontierBatched =
            operandsToValueSet(candidateBatchableOperands);
        llvm::DenseSet<mlir::Value> frontierNonBatched =
            operandsToValueSet(candidateNonBatchableOperands);

        // Predicate for all values that a batchable operand depends
        // on
        auto batchableOperandProducerPredicate = [](mlir::Value v) {
          mlir::Operation *definingOp = v.getDefiningOp();

          // Skip operations with regions so that
          // tests always only need to be
          // performed upwards and never have to
          // descend
          return (!definingOp || (mlir::isPure(definingOp) &&
                                  definingOp->getNumRegions() == 0)) &&
                 !valueIsRegionIterArg(v);
        };

        // Predicate for all values that a non-batchable operand depends
        // on
        auto nonBatchableOperandProducerPredicate = [](mlir::Value v,
                                                       mlir::scf::ForOp forOp) {
          mlir::Operation *definingOp = v.getDefiningOp();

          return isHoistable(v, forOp) &&
                 (!definingOp || definingOp->getNumRegions() == 0);
        };

        // Check that predicates hold for the initial frontiers
        if (!llvm::all_of(frontierBatched, batchableOperandProducerPredicate) ||
            !llvm::all_of(frontierNonBatched, [&](mlir::Value v) {
              return nonBatchableOperandProducerPredicate(v, innermostFor);
            })) {
          continue;
        }

        // Walk up the loop nest to find the outermost loop that
        // satisfies the conditions for the batchable and non-batchable
        // operands.
        llvm::DenseSet<mlir::Value> candidateVisitedBatched;
        llvm::DenseSet<mlir::Value> candidateVisitedNonBatched;
        llvm::SmallVector<mlir::scf::ForOp> revNest;

        int64_t candidateBatchSize = 1;
        for (mlir::scf::ForOp forOp = innermostFor;
             forOp && isCandidateLoop(forOp);
             forOp = llvm::dyn_cast<mlir::scf::ForOp>(forOp->getParentOp())) {

          int64_t thisTripCount = getStaticTripCount(forOp);

          if (maxBatchSize / candidateBatchSize < thisTripCount)
            break;
          else
            candidateBatchSize *= thisTripCount;

          std::optional<llvm::DenseSet<mlir::Value>> nextFrontierBatched =
              extendFrontier(frontierBatched, candidateVisitedBatched, forOp,
                             batchableOperandProducerPredicate);

          if (!nextFrontierBatched.has_value())
            break;

          // non-batchable operands must be defined outside or hoistable
          std::optional<llvm::DenseSet<mlir::Value>> nextFrontierNonBatched =
              extendFrontier(frontierNonBatched, candidateVisitedNonBatched,
                             forOp, [&](mlir::Value v) {
                               return nonBatchableOperandProducerPredicate(
                                   v, forOp);
                             });

          if (!nextFrontierNonBatched.has_value())
            break;

          frontierBatched = nextFrontierBatched.value();
          frontierNonBatched = nextFrontierNonBatched.value();

          revNest.push_back(forOp);
        }

        // Skip if no loop nest satisfying constraints had been found
        if (revNest.size() == 0)
          continue;

        int64_t revBatchSize = getNestedStaticTripCount(revNest);

        // Skip empty loop nests
        if (revBatchSize == 0)
          continue;

        nest = llvm::to_vector(llvm::reverse(revNest));
        batchSize = revBatchSize;
        targetOp = scalarOp;
        variant = candidateVariant;
        batchableOperands = std::move(candidateBatchableOperands);
        nonBatchableOperands = std::move(candidateNonBatchableOperands);
        visitedBatched = std::move(candidateVisitedBatched);
        visitedNonBatched = std::move(candidateVisitedNonBatched);

        return mlir::WalkResult::interrupt();
      }

      return mlir::WalkResult::skip();
    });

    // if no suitable batchable operation was found, bail out
    if (!targetOp)
      return mlir::failure();

    mlir::scf::ForOp outermostFor = nest[0];
    rewriter.setInsertionPoint(outermostFor);

    mlir::SmallVector<mlir::Value> iterArgs;

    // Create a tensor for each batchable operand with the right size
    // to be used as loop-carried dependencies in a loop nest
    // collecting the input values of the batched operation
    for (mlir::OpOperand *batchableOperand : batchableOperands) {
      mlir::Type nonBatchedType = batchableOperand->get().getType();
      llvm::SmallVector<int64_t> batchedShape =
          map(nest,
              static_cast<int64_t (*)(mlir::scf::ForOp)>(&getStaticTripCount));
      mlir::Type batchedElementType;

      if (mlir::RankedTensorType nonBatchedTensorType =
              nonBatchedType.dyn_cast<mlir::RankedTensorType>()) {
        batchedShape.append(nonBatchedTensorType.getShape().begin(),
                            nonBatchedTensorType.getShape().end());
        batchedElementType = nonBatchedTensorType.getElementType();
      } else {
        batchedElementType = nonBatchedType;
      }

      mlir::RankedTensorType batchedTensorType =
          RankedTensorType::get(batchedShape, batchedElementType);

      // TODO check that there are no dynamic sizes
      mlir::Value iterArg = rewriter.create<mlir::bufferization::AllocTensorOp>(
          outermostFor.getLoc(), batchedTensorType, mlir::ValueRange{});

      iterArgs.push_back(iterArg);
    }

    // Now reconstruct the loop nest with normalized IV ranges and
    // clone all necessary operations
    mlir::IRMapping mapping;
    llvm::SmallVector<mlir::scf::ForOp> clonedNest;

    // Create a loop nest producing the tensors for the batched input
    // operands. The operations producing the individual scalar values
    // are cloned from the original loop nest containing the batchable
    // operation.
    for (mlir::scf::ForOp forOp : nest) {
      mlir::scf::ForOp clonedForOp = rewriter.create<mlir::scf::ForOp>(
          forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(), iterArgs);
      clonedNest.push_back(clonedForOp);

      iterArgs =
          llvm::to_vector_of<mlir::Value>(clonedForOp.getRegionIterArgs());

      rewriter.setInsertionPoint(clonedForOp.getBody(),
                                 clonedForOp.getBody()->begin());

      mapping.map(forOp.getInductionVar(), clonedForOp.getInductionVar());

      for (mlir::Operation &op : forOp.getBody()->getOperations()) {
        if (llvm::any_of(op.getResults(), [&](mlir::Value res) {
              return visitedBatched.find(res) != visitedBatched.end() ||
                     visitedNonBatched.find(res) != visitedNonBatched.end();
            })) {
          rewriter.clone(op, mapping);
        }
      }
    }

    // Build the index for the batchable operands in their tensors
    mlir::ImplicitLocOpBuilder ilob(outermostFor.getLoc(), rewriter);
    llvm::SmallVector<mlir::Value> idx =
        buildNormalizedIndexes(rewriter, clonedNest);

    // Build the operations that insert the scalar values of the
    // batches into the respective tensors
    llvm::SmallVector<mlir::Value> yields;
    for (auto batchableOperandAndIterArg :
         llvm::zip(batchableOperands, iterArgs)) {
      mlir::Value batchableOperand =
          std::get<0>(batchableOperandAndIterArg)->get();
      mlir::Value iterArg = std::get<1>(batchableOperandAndIterArg);

      if (mlir::RankedTensorType batchElementType =
              llvm::dyn_cast<mlir::RankedTensorType>(
                  batchableOperand.getType())) {
        llvm::SmallVector<OpFoldResult> offsets =
            map(idx, getValueAsOpFoldResult);
        llvm::SmallVector<OpFoldResult> strides(idx.size(),
                                                ilob.getI64IntegerAttr(1));
        llvm::SmallVector<OpFoldResult> sizes(idx.size(),
                                              ilob.getI64IntegerAttr(1));

        offsets.append(batchElementType.getShape().size(),
                       ilob.getI64IntegerAttr(0));
        strides.append(batchElementType.getShape().size(),
                       ilob.getI64IntegerAttr(1));

        for (int64_t dim : batchElementType.getShape()) {
          sizes.push_back(ilob.getI64IntegerAttr(dim));
        }

        mlir::Value updatedBatchedVector =
            ilob.create<mlir::tensor::InsertSliceOp>(
                mapping.lookupOrDefault(batchableOperand), iterArg, offsets,
                sizes, strides);
        yields.push_back(updatedBatchedVector);
      } else {
        mlir::Value updatedBatchedVector = ilob.create<mlir::tensor::InsertOp>(
            mapping.lookupOrDefault(batchableOperand), iterArg, idx);
        yields.push_back(updatedBatchedVector);
      }
    }

    for (mlir::scf::ForOp forOp : llvm::reverse(clonedNest)) {
      ilob.setInsertionPointToEnd(forOp.getBody());

      ilob.create<mlir::scf::YieldOp>(yields);
      yields = llvm::to_vector_of<mlir::Value>(forOp.getResults());
    }

    // Hoist all non-batchable operands out of the loop nest
    llvm::SmallVector<mlir::Value> hoistedNonBatchableValues;

    for (mlir::OpOperand *nonBatchableOperand : nonBatchableOperands) {
      hoistedNonBatchableValues.push_back(
          hoistPure(rewriter, outermostFor, nonBatchableOperand->get()));
    }

    ilob.setInsertionPoint(outermostFor);

    // Flatten all batched tensors before passing them to the batched
    // operation
    llvm::SmallVector<mlir::Value> batchedOperands;

    for (auto batchableOperandAndBatchedOperand :
         llvm::zip(batchableOperands, yields)) {
      mlir::OpOperand *batchableOperand =
          std::get<0>(batchableOperandAndBatchedOperand);
      mlir::Value structuredBatchedOperand =
          std::get<1>(batchableOperandAndBatchedOperand);

      unsigned trailingDimensions = 0;

      if (mlir::RankedTensorType batchedTensorType =
              batchableOperand->get()
                  .getType()
                  .dyn_cast<mlir::RankedTensorType>()) {

        trailingDimensions = batchedTensorType.getShape().size();
      }

      batchedOperands.push_back(
          flattenTensor(ilob, structuredBatchedOperand, trailingDimensions));
    }

    // Created the actual batched operation through the op interface
    mlir::Value batchedResult = targetOp.createBatchedOperation(
        variant, ilob, batchedOperands, hoistedNonBatchableValues);

    mlir::RankedTensorType batchedResultType =
        llvm::dyn_cast<mlir::RankedTensorType>(batchedResult.getType());

    assert(batchedResultType);

    // Recreate the original shape of the batched results with the
    // normalized dimensions of the original loop nest
    llvm::SmallVector<int64_t> structuredBatchedShape = map(
        nest, static_cast<int64_t (*)(mlir::scf::ForOp)>(&getStaticTripCount));
    if (batchedResultType.getShape().size() > 1) {
      structuredBatchedShape.push_back(
          batchedResultType
              .getShape()[batchedResultType.getShape().size() - 1]);
    }

    mlir::RankedTensorType structuredBatchedResultType =
        mlir::RankedTensorType::get(structuredBatchedShape,
                                    batchedResultType.getElementType());

    mlir::Value structuredBatchedResult =
        unflattenTensor(ilob, batchedResult, structuredBatchedResultType);

    // Replace the original batchable operation with an operation that
    // extracts the respective scalar result from the batch of results
    // produced by the batched operation
    mlir::ImplicitLocOpBuilder ilob2(targetOp.getLoc(), rewriter);
    llvm::SmallVector<mlir::Value> idxUse =
        buildNormalizedIndexes(rewriter, nest);
    rewriter.setInsertionPoint(targetOp);

    if (batchedResultType.getShape().size() == 1) {
      rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(
          targetOp, structuredBatchedResult, idxUse);
    } else {
      llvm::SmallVector<OpFoldResult> offsets =
          map(idxUse, getValueAsOpFoldResult);
      llvm::SmallVector<OpFoldResult> strides(1, ilob2.getI64IntegerAttr(1));
      llvm::SmallVector<OpFoldResult> sizes(1, ilob2.getI64IntegerAttr(1));

      offsets.append(batchedResultType.getShape().size() - 1,
                     ilob2.getI64IntegerAttr(0));
      strides.append(batchedResultType.getShape().size() - 1,
                     ilob2.getI64IntegerAttr(1));

      for (int64_t dim : batchedResultType.getShape().take_front()) {
        strides.push_back(ilob2.getI64IntegerAttr(dim));
        sizes.push_back(ilob2.getI64IntegerAttr(dim));
      }

      rewriter.replaceOpWithNewOp<mlir::tensor::ExtractSliceOp>(
          targetOp, batchedResult, offsets, sizes, strides);
    }

    return mlir::success();
  }

private:
  int64_t maxBatchSize;
};

// Returns a pair containing:
//
//  - the set of loops whose IVs are referenced in the indexing
//    dimensions of `op` and which only appear in pure quasi-affine
//    expressions with a constant step wrt. to the iteration space and
//    where the step is equal to the size times the offset of the
//    dimension indexed by the expression.
//
//  - an array defining the order in which these loop IVs are
//    referenced in the indexes
//
template <typename IndexedOpTy>
std::pair<llvm::DenseSet<mlir::scf::ForOp>, llvm::SmallVector<mlir::scf::ForOp>>
getLoopsForCandidateIndexes(IndexedOpTy op) {
  llvm::DenseSet<mlir::scf::ForOp> allIVs;
  llvm::DenseSet<mlir::scf::ForOp> qaIVs;
  llvm::SmallVector<mlir::scf::ForOp> orderedQAIVs;

  for (auto it : llvm::enumerate(IndexedOpInfo<IndexedOpTy>::getOffsets(op))) {
    mlir::OpFoldResult expr = it.value();
    size_t dimIdx = it.index();

    if (mlir::Value dynExpr = expr.dyn_cast<mlir::Value>()) {
      walkUseDefChain(dynExpr, [&](mlir::Value v) {
        if (auto loop = valueIsRegionIterArg(v))
          allIVs.insert(*loop);
      });
    }

    mlir::scf::ForOp qaLoop;
    LoopsBoundsAndStep bas;

    if (isQuasiAffineIVExpressionWithConstantStep(expr, &qaLoop, &bas)) {
      if (qaLoop) {
        int64_t sliceExtents = getSliceExtents(op, dimIdx);

        if (sliceExtents == 1 || sliceExtents == bas.step) {
          qaIVs.insert(qaLoop);
          orderedQAIVs.push_back(qaLoop);
        }
      }
    }
  }

  llvm::DenseSet<mlir::scf::ForOp> res = setMinus(qaIVs, allIVs);
  llvm::SmallVector<mlir::scf::ForOp> orderedRes =
      filterVector(orderedQAIVs, res);

  return std::make_pair(std::move(res), std::move(orderedRes));
}

/// Cleanup pattern that replaces a chain of a `tensor.extract` /
/// `tensor.extract_slice`, a `tensor.insert` / `tensor.insert_slice` and an
/// `scf.yield` op with `tensor.extract_slice` and `tensor.insert_slice` ops.
/// E.g.,
///
///   %res0 = scf.for %i = c0 to %cN step %c1 iter_args(%arg0 = %T1) {
///     %res1 = scf.for %j = c0 to %cM step %c1 iter_args(%arg1 = %arg0) {
///       %res2 = scf.for %k = c0 to %cK step %c1 iter_args(%arg2 = %arg1) {
///         %s = tensor.extract %T2[%i, %j, %k]
///         %TRes = tensor.insert %s into %arg2
///         scf.yield %arg2
///       }
///       scf.yield %res2
///     }
///     scf.yield %res1
///   }
///
/// is replaced with:
///
///   %tmp = tensor.extract_slice %T2
///   %res0 = tensor.insert_slice %tmp into %T1
///
/// Any index may be a quasi-affine expression on a single loop
/// induction variable, but the distance between the result for any
/// two successive values of the IV must be constant.
template <typename ExtractOpTy, typename InsertOpTy>
class CleanupPattern : public mlir::OpRewritePattern<mlir::func::FuncOp> {
public:
  CleanupPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::func::FuncOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp func,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::scf::ForOp outermostFor;
    mlir::scf::ForOp innermostFor;
    ExtractOpTy extractOp;
    InsertOpTy insertOp;
    unsigned yieldOperandNumber;
    mlir::Value dstTensor;

    func.walk([&](ExtractOpTy currExtractOp) {
      // Check that we deal with only static sizes and strides;
      // otherwise the conservative hoisting implementation would fail
      if (!IndexedOpInfo<ExtractOpTy>::hasAllStaticSizesAndStrides(
              currExtractOp)) {
        return mlir::WalkResult::skip();
      }

      // First check that the extract op is embedded in a for loop
      mlir::scf::ForOp currInnermostFor =
          llvm::dyn_cast<mlir::scf::ForOp>(currExtractOp->getParentOp());

      if (!currInnermostFor)
        return mlir::WalkResult::skip();

      if (!currInnermostFor.isDefinedOutsideOfLoop(
              IndexedOpInfo<ExtractOpTy>::getTensor(currExtractOp))) {
        return mlir::WalkResult::skip();
      }

      mlir::scf::YieldOp yield = llvm::dyn_cast<mlir::scf::YieldOp>(
          currInnermostFor.getBody()->getTerminator());

      // Next, find a chain of the 3 relevant operations:
      //
      //  %s  = tensor.extract %T[...] (or tensor.extract_slice)
      //  %U' = tensor.insert %s into %U[...] (or tensor.insert_slice)
      //  scf.yield %U'
      //
      // with all affine indexes. The indexes used by the extract must
      // be a suffix of the indexes of the insert op and the tensor
      // that is updated must not be derived from the tensor from
      // which the element is extracted.
      InsertOpTy currInsertOp;

      std::pair<llvm::DenseSet<mlir::scf::ForOp>,
                llvm::SmallVector<mlir::scf::ForOp>>
          qaLoopsExtract = getLoopsForCandidateIndexes(currExtractOp);

      for (mlir::OpOperand &extractUser : currExtractOp->getUses()) {
        if (InsertOpTy currInsertOp =
                llvm::dyn_cast<InsertOpTy>(extractUser.getOwner())) {
          // Insert and extract must be in the same loop
          if (currInsertOp->getParentOp() != currInnermostFor.getOperation())
            continue;

          // Insert op must also have static sizes and strides in
          // order to be hoistable
          if (!IndexedOpInfo<InsertOpTy>::hasAllStaticSizesAndStrides(
                  currInsertOp)) {
            continue;
          }

          if (!isSoleUser(currInsertOp.getResult(), yield))
            continue;

          // Insertion must be into a tensor that is a region
          // iteration argument and must be the only user of that
          // value (since it will be erased upon hoisting)
          if (!valueIsRegionIterArgOf(currInsertOp.getDest(),
                                      currInnermostFor) ||
              !isSoleUser(currInsertOp.getDest(), currInsertOp))
            continue;

          std::pair<llvm::DenseSet<mlir::scf::ForOp>,
                    llvm::SmallVector<mlir::scf::ForOp>>
              qaLoopsInsert = getLoopsForCandidateIndexes(currInsertOp);

          llvm::DenseSet<mlir::scf::ForOp> qaLoopsBoth =
              intersectSets(qaLoopsExtract.first, qaLoopsInsert.first);

          if (qaLoopsBoth.size() == 0)
            continue;

          if (!qaLoopsBoth.contains(currInnermostFor))
            continue;

          // Indexes must appear in the same order and the same number
          // of times, such that the extracted and inserted slices
          // after the cleanup have the same shape and order
          llvm::SmallVector<mlir::scf::ForOp> orderedLoopsExtract =
              filterVector(qaLoopsExtract.second, qaLoopsBoth);
          llvm::SmallVector<mlir::scf::ForOp> orderedLoopsInsert =
              filterVector(qaLoopsInsert.second, qaLoopsBoth);

          if (orderedLoopsExtract.size() != orderedLoopsInsert.size() ||
              orderedLoopsExtract != orderedLoopsInsert) {
            continue;
          }

          mlir::scf::ForOp candidateOutermostFor;
          mlir::scf::ForOp candidateInnermostFor;

          unsigned currYieldOperandNumber =
              getOperandIndexForValue(yield, currInsertOp.getResult());

          getLongestPerfectLoopnest(
              qaLoopsBoth, candidateInnermostFor, candidateOutermostFor,
              [&](mlir::scf::ForOp parent, mlir::scf::ForOp child) {
                // Check that the nest from the outermost to
                // the innermost loop is perfect and forwards
                // the result of the innermost loop to the
                // outermost one

                mlir::scf::YieldOp parentYield =
                    llvm::dyn_cast<mlir::scf::YieldOp>(
                        parent.getBody()->getTerminator());

                if (!parent.isDefinedOutsideOfLoop(
                        IndexedOpInfo<ExtractOpTy>::getTensor(currExtractOp)))
                  return false;

                return parentYield.getOperand(currYieldOperandNumber) ==
                       child.getResult(currYieldOperandNumber);
              });

          insertOp = currInsertOp;
          yieldOperandNumber = currYieldOperandNumber;
          outermostFor = candidateOutermostFor;
          innermostFor = currInnermostFor;
          extractOp = currExtractOp;
          dstTensor =
              candidateOutermostFor.getInitArgs()[currYieldOperandNumber];

          return mlir::WalkResult::interrupt();
        }
      }

      return mlir::WalkResult::skip();
    });

    if (!outermostFor)
      return mlir::failure();

    mlir::Value slice = hoistExtractOp(rewriter, outermostFor, extractOp);

    mlir::Value insertedSlice =
        hoistInsertOp(rewriter, slice, dstTensor, outermostFor, insertOp);

    rewritePerfectLoopNestWithReplacedNthResult(
        rewriter, innermostFor, outermostFor, yieldOperandNumber,
        insertedSlice);

    return mlir::success();
  }
};

// Folds the operation `op` by recursively folding all
// producers. Occurrences of `arg` are replaced with `argVal`.
// All encountered operations must produce a single result.
static mlir::Attribute fold(mlir::Operation *op, mlir::Value arg,
                            mlir::Attribute argVal) {
  assert(op->getNumResults() == 1);

  // Check if `arg` needs to be replaced with `argVal`
  if (op->getResult(0) == arg)
    return argVal;

  // Constants are just folded to their value attributes
  if (mlir::arith::ConstantOp cstOp =
          llvm::dyn_cast<mlir::arith::ConstantOp>(op)) {
    return cstOp.getValue();
  }

  // Recursively fold all producers and collect the folding results
  // for each operand
  llvm::SmallVector<mlir::Attribute> foldedOperands;

  for (mlir::OpOperand &operand : op->getOpOperands()) {
    mlir::Operation *producer = operand.get().getDefiningOp();
    assert(producer);

    foldedOperands.push_back(fold(producer, arg, argVal));
  }

  // Invoke the folder for this operation
  llvm::SmallVector<mlir::OpFoldResult> res;
  mlir::LogicalResult foldRes = op->fold(foldedOperands, res);

  assert(foldRes.succeeded());
  assert(res.size() == 1);

  mlir::Attribute resAttr = res[0].dyn_cast<mlir::Attribute>();

  assert(resAttr);

  return resAttr;
}

// Folding pattern that collapses operations on constant dense tensors
// into a new constant. E.g.,
//
//   %cst = arith.constant dense<...> : tensor<Nxi9>
//   %res = scf.for %i = %c0 to %cN {
//     %cst_i9 = tensor.extract %cst[%i]
//     %cst_i64 = arith.extui %cst_i9 : i64
//     ...
//   }
//
// becomes:
//
//   %cst = arith.constant dense<...> : tensor<Nxi64>
//   %res = scf.for %i = %c0 to %cN {
//     %cst_i64 = tensor.extract %cst[%i]
//     ...
//   }
class ConstantDenseFoldingPattern
    : public mlir::OpRewritePattern<mlir::func::FuncOp> {
protected:
  // Checks if an operation is foldable
  static bool isFoldableOp(mlir::Operation *op) {
    return op->getNumResults() == 1 && mlir::isPure(op) &&
           llvm::TypeSwitch<mlir::Operation *, bool>(op)
               .Case<mlir::arith::AddIOp, mlir::arith::ExtSIOp,
                     mlir::arith::ConstantOp>([](auto op) { return true; })
               .Default([](auto op) { return false; });
  }

  // Checks if `v` can be calculated statically given that the values
  // in `foldables` are static. The function recursively collects all
  // intermediate values which have been found to be static in
  // `foldables`.
  static bool isFoldableValue(mlir::Value v,
                              llvm::DenseSet<mlir::Value> &foldables) {
    if (foldables.contains(v))
      return true;

    mlir::Operation *op = v.getDefiningOp();

    if (!op || !isFoldableOp(op))
      return false;

    if (llvm::all_of(op->getOperands(), [&](mlir::Value v) {
          return isFoldableValue(v, foldables);
        })) {
      for (mlir::Value v : op->getOperands())
        foldables.insert(v);

      return true;
    }

    return false;
  }

  // Generates a flat index from a tensor with the shape `shape`
  // indexed by `idx`
  static int64_t linearizeIndex(llvm::ArrayRef<int64_t> shape,
                                llvm::ArrayRef<int64_t> idx) {
    int64_t flatIdx = 0;
    int64_t mul = 1;
    int64_t n = shape.size();

    for (int64_t i = 0; i < n; i++) {
      flatIdx += mul * idx[n - i - 1];
      mul *= shape[n - i - 1];
    }

    return flatIdx;
  };

public:
  ConstantDenseFoldingPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::func::FuncOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp func,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::tensor::ExtractOp extractOp;
    mlir::Operation *targetOp = nullptr;
    llvm::SmallVector<mlir::scf::ForOp> nest;
    llvm::SmallVector<mlir::scf::ForOp> idxMap;
    llvm::SmallVector<LoopsBoundsAndStep> idxBounds;
    mlir::arith::ConstantOp cdo;
    mlir::RankedTensorType constantType;
    mlir::Type origElementType;
    mlir::Type foldedElementType;

    func.walk([&](mlir::tensor::ExtractOp currExtractOp) {
      // Check that the extraction in on a value produced by an
      // `arith.constant_dense` operation.
      mlir::arith::ConstantOp currCdo =
          llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
              currExtractOp.getTensor().getDefiningOp());

      if (!currCdo)
        return mlir::WalkResult::skip();

      if (!isSoleUser(currCdo.getResult(), currExtractOp))
        return mlir::WalkResult::skip();

      mlir::RankedTensorType currConstantType =
          currExtractOp.getTensor()
              .getType()
              .dyn_cast<mlir::RankedTensorType>();

      if (!currConstantType)
        return mlir::WalkResult::skip();

      // First check that the extract op is embedded in a for loop
      mlir::scf::ForOp currInnermostFor =
          llvm::dyn_cast<mlir::scf::ForOp>(currExtractOp->getParentOp());

      if (!currInnermostFor)
        return mlir::WalkResult::skip();

      llvm::DenseSet<mlir::scf::ForOp> nestUnsorted;
      llvm::SmallVector<mlir::scf::ForOp> currIdxMap;
      llvm::SmallVector<LoopsBoundsAndStep> currIdxBounds;

      // Make sure that the extract operation uses only quasi affine
      // expressions on IVs, where each index uses at most a single
      // IV.
      for (mlir::Value idx : currExtractOp.getIndices()) {
        mlir::scf::ForOp forOp;
        LoopsBoundsAndStep bas;

        if (!isQuasiAffineIVExpressionWithConstantStep(idx, &forOp, &bas))
          return mlir::WalkResult::skip();

        if (forOp)
          nestUnsorted.insert(forOp);

        currIdxBounds.push_back(bas);
        currIdxMap.push_back(forOp);
      }

      llvm::DenseSet<mlir::Value> foldables;
      foldables.insert(currExtractOp.getResult());

      if (!currExtractOp.getResult().hasOneUse())
        return mlir::WalkResult::skip();

      mlir::Operation *firstUser =
          currExtractOp.getResult().getUses().begin()->getOwner();
      mlir::Operation *currOp = firstUser;
      mlir::Operation *currTargetOp = nullptr;
      mlir::Type currOrigElementType = currConstantType.getElementType();
      mlir::Type currFoldedElementType = currOrigElementType;

      // Walk down the def-use chain from the extract operation until
      // an operation is found that is not foldable
      while (true) {
        if (!isFoldableOp(currOp))
          break;

        if (currOp->getNumResults() != 1 || !currOp->getResult(0).hasOneUse() ||
            currOp->getParentOp() != currInnermostFor.getOperation())
          break;

        if (!llvm::all_of(currOp->getOperands(), [&](mlir::Value v) {
              return isFoldableValue(v, foldables);
            }))
          break;

        currFoldedElementType = currOp->getResult(0).getType();

        currTargetOp = currOp;
        currOp = currOp->getUses().begin()->getOwner();
      }

      if (!currTargetOp)
        return mlir::WalkResult::skip();

      // Check constraints on the index space of the extract
      // operation. I.e., if the type changes during the folding,
      // ensure that the index space covers the entire tensor and that
      // there are no out-of-bounds accesses.
      for (auto it : llvm::enumerate(currExtractOp.getIndices())) {
        mlir::scf::ForOp forOp;
        LoopsBoundsAndStep bas;
        mlir::Value idx = it.value();
        size_t i = it.index();

        if (!isQuasiAffineIVExpressionWithConstantStep(idx, &forOp, &bas))
          return mlir::WalkResult::skip();

        // If the type changes by the folding, the entire tensor needs
        // to be rewritten
        if (currFoldedElementType != currOrigElementType) {
          if (bas.lb != 0 || bas.ub != currConstantType.getDimSize(i) ||
              bas.step != 1)
            return mlir::WalkResult::skip();
        }
        // Otherwise, just make sure that there are no out-of-bounds
        // accesses
        else {
          if (bas.ub - bas.step >= currConstantType.getDimSize(i))
            return mlir::WalkResult::skip();
        }
      }

      extractOp = currExtractOp;
      targetOp = currTargetOp;

      nest = sortLoopsOutermostToInnermost(nestUnsorted);
      idxMap = std::move(currIdxMap);
      idxBounds = std::move(currIdxBounds);
      cdo = currCdo;
      constantType = currConstantType;
      origElementType = currOrigElementType;
      foldedElementType = currFoldedElementType;

      return mlir::WalkResult::interrupt();
    });

    if (!targetOp)
      return mlir::failure();

    // Original tensor of constants
    auto denseVals = cdo.getValueAttr()
                         .cast<mlir::DenseElementsAttr>()
                         .getValues<mlir::Attribute>();

    // Updated tensor of constants initialized with original values
    SmallVector<mlir::Attribute> newDenseVals(denseVals.begin(),
                                              denseVals.end());

    mlir::SmallVector<int64_t> tripCounts = map(
        nest, [](mlir::scf::ForOp forOp) { return getStaticTripCount(forOp); });

    // Number of iterations already performed for each loop
    mlir::SmallVector<int64_t> trips(nest.size(), 0);

    // current index
    mlir::SmallVector<int64_t> idx =
        map(idxBounds, [](LoopsBoundsAndStep &bas) { return bas.lb; });

    // Maps the index of each IV in the loop nest to the indexes of
    // the extract operation
    mlir::SmallVector<mlir::SmallVector<size_t>> revIdxMap(nest.size());

    for (size_t i = 0; i < idxMap.size(); i++) {
      for (size_t j = 0; j < nest.size(); j++) {
        if (nest[j] == idxMap[i]) {
          revIdxMap[j].push_back(i);
          break;
        }
      }
    }

    size_t i = nest.size() - 1;

    // Reset the trip count for a loop back to zero and reinitializes
    // all indexes using the associated IV
    auto resetTrips = [&](size_t loopIdx) {
      trips[loopIdx] = 0;

      for (size_t i : revIdxMap[loopIdx]) {
        idx[i] = idxBounds[i].lb;
      }
    };

    // Increases the trip count of a loop by one and calculates the
    // next value of all indexes using the associated IV
    auto incTrips = [&](size_t loopIdx) {
      trips[loopIdx] += 1;

      for (size_t i : revIdxMap[loopIdx]) {
        idx[i] += idxBounds[i].step;
      }
    };

    // Iterate over the entire iteration space of the loop nest. The
    // variable i represents the index of the loop that is currently
    // stepped in the nest
    while (true) {
      // Loop has reached its maximum trip count. If the loop ist the
      // first in the nest, the entire space has been
      // covered. Otherwise, reset the trip count of the current loop
      // and step the loop above.
      if (trips[i] == tripCounts[i]) {
        if (i == 0)
          break;

        resetTrips(i);
        i--;

        incTrips(i);
      } else {
        // Trip count of the current loop hasn't been reached. If this
        // is the innermost loop, calculate a new index, fold all
        // values and write the result to the new tensor of
        // constants. Otherwise, switch to the next loop in the nest.
        if (i == nest.size() - 1) {
          size_t flatIdx = linearizeIndex(constantType.getShape(), idx);

          mlir::Attribute newVal =
              fold(targetOp, extractOp.getResult(), denseVals[flatIdx]);

          newDenseVals[flatIdx] = newVal;
          incTrips(i);
        } else {
          i++;
        }
      }
    }

    // Create a new `arith.constant` operation with the updated tensor
    // of constants
    mlir::DenseElementsAttr newDenseElementsAttr = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get(constantType.getShape(), foldedElementType),
        newDenseVals);

    rewriter.setInsertionPoint(cdo);
    mlir::arith::ConstantOp newCdo = rewriter.create<mlir::arith::ConstantOp>(
        cdo.getLoc(), newDenseElementsAttr);

    rewriter.setInsertionPoint(targetOp);

    // Replace the last op in the chain of foldable operations with a
    // `tensor.extract` op on the new tensor of constants.
    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(
        targetOp, targetOp->getResult(0).getType(), newCdo,
        extractOp.getIndices());

    return mlir::success();
  }
};

class TensorAllocationCleanupPattern
    : public mlir::OpRewritePattern<mlir::func::FuncOp> {
public:
  TensorAllocationCleanupPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::func::FuncOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp func,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::LogicalResult res = mlir::failure();

    func.walk([&](mlir::bufferization::AllocTensorOp allocOp) {
      if (allocOp->getUses().empty()) {
        rewriter.eraseOp(allocOp);
        res = mlir::success();
      }
    });

    return res;
  }
};

class BatchingPass : public BatchingBase<BatchingPass> {
public:
  BatchingPass(int64_t maxBatchSize) : maxBatchSize(maxBatchSize) {}
  void runOnOperation() override {
    mlir::Operation *op = getOperation();

    mlir::RewritePatternSet patterns(op->getContext());
    patterns.add<BatchingPattern>(op->getContext(), maxBatchSize);
    patterns
        .add<CleanupPattern<mlir::tensor::ExtractOp, mlir::tensor::InsertOp>,
             CleanupPattern<mlir::tensor::ExtractSliceOp,
                            mlir::tensor::InsertSliceOp>,
             ConstantDenseFoldingPattern, TensorAllocationCleanupPattern>(
            op->getContext());

    if (mlir::applyPatternsAndFoldGreedily(op, std::move(patterns)).failed())
      this->signalPassFailure();
  }

private:
  int64_t maxBatchSize;
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createBatchingPass(int64_t maxBatchSize) {
  return std::make_unique<BatchingPass>(maxBatchSize);
}

} // namespace concretelang
} // namespace mlir
