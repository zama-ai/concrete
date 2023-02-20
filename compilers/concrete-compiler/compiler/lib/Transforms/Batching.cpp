// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <concretelang/Interfaces/BatchableInterface.h>
#include <concretelang/Transforms/Passes.h>

namespace mlir {
namespace concretelang {
/// Checks if `forOp` has constant bounds and a constant step.
static bool isStaticLoop(mlir::scf::ForOp forOp, int64_t *ilb = nullptr,
                         int64_t *iub = nullptr, int64_t *istep = nullptr) {

  mlir::Operation *lbOp = forOp.getLowerBound().getDefiningOp();
  mlir::Operation *ubOp = forOp.getUpperBound().getDefiningOp();
  mlir::Operation *stepOp = forOp.getStep().getDefiningOp();

  if (!lbOp || !ubOp || !stepOp)
    return false;

  mlir::arith::ConstantIndexOp lb =
      llvm::dyn_cast<mlir::arith::ConstantIndexOp>(lbOp);
  mlir::arith::ConstantIndexOp ub =
      llvm::dyn_cast<mlir::arith::ConstantIndexOp>(ubOp);
  mlir::arith::ConstantIndexOp step =
      llvm::dyn_cast<mlir::arith::ConstantIndexOp>(stepOp);

  if (lb && ub && step) {
    if (ilb)
      *ilb = lb.value();

    if (iub)
      *iub = ub.value();

    if (istep)
      *istep = step.value();

    return true;
  }

  return false;
}

/// Checks if `forOp` is a loop with a lower bound of 0, a constant
/// upper bound and a constant step of 1
static bool isStaticNormalizedLoop(mlir::scf::ForOp forOp) {
  int64_t lb;
  int64_t step;

  if (isStaticLoop(forOp, &lb, nullptr, &step))
    return (lb == 0 && step == 1);

  return false;
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

/// Assumes that `v` is a constant index operation and returns the
/// constant value as an `int64_t`.
static int64_t getConstantIndexValue(mlir::Value v) {
  assert(v.getDefiningOp() &&
         llvm::isa<mlir::arith::ConstantIndexOp>(*v.getDefiningOp()));

  return llvm::dyn_cast<mlir::arith::ConstantIndexOp>(*v.getDefiningOp())
      .value();
}

/// Returns a `Value` from an `OpFoldResult`. If the `OpFoldResult` is
/// a already a value, the value is returned as is. Otherwise a
/// constant op is created using `builder`.
static mlir::Value getOpFoldResultAsValue(mlir::ImplicitLocOpBuilder &builder,
                                          mlir::OpFoldResult v) {
  if (v.is<mlir::Value>()) {
    return v.dyn_cast<mlir::Value>();
  } else {
    return builder.create<mlir::arith::ConstantIndexOp>(
        v.get<mlir::Attribute>().cast<mlir::IntegerAttr>().getInt());
  }
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

/// Helper class whose call operator compares its argument to the
/// constant value `cst`.
template <typename T, const T cst> struct comparator {
  bool operator()(const T &val) { return cst == val; }
};

/// Divides `a` by `b`, where both values can be any combination of
/// `IntegerAttr` and `Value`.
static mlir::OpFoldResult opFoldDiv(mlir::ImplicitLocOpBuilder &builder,
                                    mlir::OpFoldResult a,
                                    mlir::OpFoldResult b) {
  return opFoldExpr<mlir::arith::DivSIOp, std::divides<int64_t>,
                    comparator<int64_t, 1>>(builder, a, b);
}

/// Subtracts `b` from `a`, where both values can be any combination
/// of `IntegerAttr` and `Value`.
static mlir::OpFoldResult opFoldSub(mlir::ImplicitLocOpBuilder &builder,
                                    mlir::OpFoldResult a,
                                    mlir::OpFoldResult b) {
  return opFoldExpr<mlir::arith::SubIOp, std::minus<int64_t>,
                    comparator<int64_t, 0>>(builder, a, b);
}

/// Convenience class that holds all parameters of a loop
struct BoundsAndStep {
  int64_t lb;
  int64_t ub;
  int64_t step;

  BoundsAndStep operator+(const BoundsAndStep &other) {
    return BoundsAndStep{lb + other.lb, ub + other.ub, step + other.step};
  }
  BoundsAndStep operator-(const BoundsAndStep &other) {
    return BoundsAndStep{lb - other.lb, ub - other.ub, step - other.step};
  }
  BoundsAndStep operator*(const BoundsAndStep &other) {
    return BoundsAndStep{lb * other.lb, ub * other.ub, step * other.step};
  }
  BoundsAndStep operator/(int64_t d) {
    return BoundsAndStep{lb / d, ub / d, step / d};
  }
};

/// Returns the lower bound, upper bound and step of the quasi-affine
/// expression `expr` on the the induction variable from a for
/// operation.
static std::optional<BoundsAndStep>
getBoundsOfQuasiAffineIVExpression(mlir::Value expr, mlir::scf::ForOp forOp) {
  // Base case: expression is the induction variable itself -> return
  // loop bounds
  if (expr == forOp.getInductionVar()) {
    return BoundsAndStep{getConstantIndexValue(forOp.getLowerBound()),
                         getConstantIndexValue(forOp.getUpperBound()),
                         getConstantIndexValue(forOp.getStep())};
  }
  // Arithmetic expression
  else if (mlir::Operation *op = expr.getDefiningOp()) {
    if (llvm::isa<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp,
                  mlir::arith::DivSIOp>(op)) {

      std::optional<BoundsAndStep> lhs =
          getBoundsOfQuasiAffineIVExpression(op->getOperand(0), forOp);
      std::optional<BoundsAndStep> rhs =
          getBoundsOfQuasiAffineIVExpression(op->getOperand(1), forOp);

      if (!lhs.has_value() || !rhs.has_value())
        return std::nullopt;

      if (llvm::isa<mlir::arith::AddIOp>(op))
        return *lhs + *rhs;
      else if (llvm::isa<mlir::arith::SubIOp>(op))
        return *lhs - *rhs;
      else if (llvm::isa<mlir::arith::MulIOp>(op))
        return (*lhs) * (*rhs);
      else if (llvm::isa<mlir::arith::DivSIOp>(op)) {
        assert(rhs->ub == rhs->lb && rhs->step == 0 &&
               "Expression for divisor references IV");
        int64_t rhsVal = rhs->ub;

        assert(rhsVal != 0 && "Division by zero");

        // If the step value of the subexpression is not a multiple of
        // the divisor, there may be two iterations with the same
        // value. Conservatively bail out.
        if (lhs->step % rhsVal != 0)
          return std::nullopt;

        return *lhs / rhsVal;
      }
    }
    // Base case: constant -> return constant value
    else if (llvm::isa<mlir::arith::ConstantIndexOp>(expr.getDefiningOp())) {
      mlir::arith::ConstantIndexOp cst =
          llvm::dyn_cast<mlir::arith::ConstantIndexOp>(expr.getDefiningOp());
      return BoundsAndStep{cst.value(), cst.value(), 0};
    }
  }

  llvm_unreachable("Expression could not be evaluated statically");
}

/// Checks whether the expression `expr` is a quasi-affine expression
/// on a single induction variable. If an induction variable is
/// referenced, the owning for loop is returned in `*owningForOp`.
static bool isQuasiAffineIVExpression(mlir::Value expr,
                                      mlir::scf::ForOp *owningForOp = nullptr) {
  if (mlir::Operation *op = expr.getDefiningOp()) {
    if (llvm::isa<mlir::arith::ConstantIndexOp>(op)) {
      return true;
    } else if (llvm::isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
                         mlir::arith::MulIOp, mlir::arith::DivSIOp>(op)) {
      mlir::scf::ForOp forLHS;
      mlir::scf::ForOp forRHS;

      if (!isQuasiAffineIVExpression(op->getOperand(0), &forLHS) ||
          !isQuasiAffineIVExpression(op->getOperand(1), &forRHS)) {
        return false;
      } else {
        // Check that appearances of IVs refer to the same IV
        if (forLHS && forRHS && forLHS != forRHS)
          return false;
      }

      // Assume that the expression is already canonicalized, so that
      // IVs appear only in numerators and on one side of a
      // multiplication subexpression
      if ((llvm::isa<mlir::arith::MulIOp>(op) && forLHS && forRHS) ||
          (llvm::isa<mlir::arith::DivSIOp>(op) && forRHS))
        return false;

      if (owningForOp != nullptr) {
        if (forLHS)
          *owningForOp = forLHS;
        else if (forRHS)
          *owningForOp = forRHS;
      }

      return true;
    }

    return false;
  }
  // Base case: Expression is an induction variable
  else if (mlir::scf::ForOp forOp = scf::getForInductionVarOwner(expr)) {
    if (owningForOp != nullptr)
      *owningForOp = forOp;

    return true;
  }

  return false;
}

/// Invokes `callback` for every subexpression of `expr` that is an
/// induction variable with the corresponding for operation as the
/// argument. Stops if the callback function returns `true`.
static void forEveryReferencedInductionVarBreakable(
    mlir::Value expr, llvm::function_ref<bool(mlir::scf::ForOp)> callback) {
  if (mlir::scf::ForOp forOp = scf::getForInductionVarOwner(expr)) {
    callback(forOp);
  } else {
    if (expr.getDefiningOp()) {
      for (mlir::Value operand : expr.getDefiningOp()->getOperands()) {
        forEveryReferencedInductionVarBreakable(operand, callback);
      }
    }
  }
}

/// Invokes `callback` for every subexpression of `expr` that is an
/// induction variable with the corresponding for operation as the
/// argument.
static void forEveryReferencedInductionVar(
    mlir::Value expr, llvm::function_ref<void(mlir::scf::ForOp)> callback) {
  forEveryReferencedInductionVarBreakable(expr,
                                          [&](mlir::scf::ForOp forOp) -> bool {
                                            callback(forOp);
                                            return false;
                                          });
}

/// Returns the loop associated to the first induction variable
/// encountered in a subexpression of `expr`.
static mlir::scf::ForOp findFirstReferencedInductionVar(mlir::Value expr) {
  mlir::scf::ForOp ret;

  forEveryReferencedInductionVarBreakable(expr,
                                          [&](mlir::scf::ForOp forOp) -> bool {
                                            ret = forOp;
                                            return true;
                                          });

  return ret;
}

/// Checks if `expr` is a quasi affine expression on a single
/// induction variable, for which the increment of the induction
/// variable with the step of the associated for loop results in a
/// constant incrementation of when evaluating the expression.
///
/// E.g., this is true for the expression `i+1` for any constant step
/// size, since `((i+step)+1) - (i+1)` is constant. This is also true
/// for `(i+5)/7` for a step size that is a multiple of `7`, but false
/// for any other step size.
static bool
isQuasiAffineIVExpressionWithConstantStep(mlir::Value expr,
                                          mlir::scf::ForOp *forOp = nullptr) {
  mlir::scf::ForOp tmpForOp;

  if (isQuasiAffineIVExpression(expr, &tmpForOp)) {
    std::optional<BoundsAndStep> bas =
        getBoundsOfQuasiAffineIVExpression(expr, tmpForOp);

    if (bas.has_value()) {
      if (forOp != nullptr)
        *forOp = tmpForOp;
      return true;
    }
  }

  return false;
}

/// Hoists a an operation embedded into a loop nest that and that is
/// indexed using quasi-affine expressions of the loops' IVs as a
/// `tensor.extract_slice` outside of the outermost loop
template <typename EltWiseOp>
mlir::Value hoistIndexedOp(
    mlir::PatternRewriter &rewriter, mlir::scf::ForOp outermostFor,
    mlir::Value tensorizedOperands, EltWiseOp eltwiseOp,
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
  mlir::ImplicitLocOpBuilder ilob(eltwiseOp.getLoc(), rewriter);

  for (mlir::Value idx : eltwiseOp.getIndices()) {
    mlir::scf::ForOp forOp;
    bool isAffine = isQuasiAffineIVExpression(idx, &forOp);

    if (isAffine && forOp) {

      std::optional<BoundsAndStep> bas =
          getBoundsOfQuasiAffineIVExpression(idx, forOp);

      assert(bas.has_value());
      assert(bas->step != 0);

      offsets.push_back(rewriter.getIndexAttr(bas->lb));
      sizes.push_back(rewriter.getIndexAttr((bas->ub - bas->lb) / bas->step));
      strides.push_back(rewriter.getIndexAttr(bas->step));

      ivIndexedDims.push_back(true);
    } else if (isAffine || outermostFor.isDefinedOutsideOfLoop(idx)) {
      offsets.push_back(getValueAsOpFoldResult(idx));
      sizes.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
      ivIndexedDims.push_back(false);
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

/// Pattern that replaces a batchable operation embedded into a loop
/// nest with the batched version of the operation, e.g.,
///
///   scf.for %i = c0 to %cN step %c1 {
///     scf.for %j = c0 to %cM step %c1 {
///       scf.for %k = c0 to %cK step %c1 {
///         %s = tensor.extract %T[%i, %j, %k]
///         %res = batchable_op %s
///         ...
///       }
///     }
///   }
///
/// is replaced with:
///
///   %batchedSlice = tensor.extract_slice
///        %T[%c0, %c0, %c0] [%cN, %cM, %cK] [%c1, %c1, %c1]
///   %flatSlice = tensor.collapse_shape %batchedSlice
///   %resTFlat = batchedOp %flatSlice
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
class BatchingPattern : public mlir::OpRewritePattern<mlir::func::FuncOp> {
public:
  BatchingPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::func::FuncOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp func,
                  mlir::PatternRewriter &rewriter) const override {
    // Operation that will be hoisted out of the loop nest and
    // replaced by the batched version of the operation
    BatchableOpInterface targetOp;

    // Extract operation producing the scalar operand of the batchable
    // operation
    mlir::tensor::ExtractOp targetExtractOp;

    // Outermost for loop of the loop nest in which the batchable op
    // is located
    mlir::scf::ForOp outermostFor;

    // Find a batchable op which is embedded into a loop nest
    func.walk([&](BatchableOpInterface scalarOp) {
      // Is producer an extract op?
      auto extractOp = llvm::dyn_cast<mlir::tensor::ExtractOp>(
          scalarOp.getBatchableOperand().get().getDefiningOp());

      if (!extractOp)
        return mlir::WalkResult::skip();

      // Is extract op embedded into a loop?
      if (!isa<mlir::scf::ForOp>(extractOp->getParentOp()))
        return mlir::WalkResult::skip();

      // Find outermost for loop whose IV is used as an index
      mlir::scf::ForOp currOutermostFor;

      for (mlir::Value idx : extractOp.getIndices()) {
        forEveryReferencedInductionVar(idx, [&](mlir::scf::ForOp forOp) {
          if (!currOutermostFor ||
              forOp.getOperation()->isAncestor(currOutermostFor)) {
            currOutermostFor = forOp;
          }
        });
      }

      if (!currOutermostFor)
        return mlir::WalkResult::skip();

      if (!currOutermostFor.isDefinedOutsideOfLoop(extractOp.getTensor()))
        return mlir::WalkResult::skip();

      // Now make sure that each index is either a quasi-affine
      // expression on a single Loop IV, with a constant offset for
      // all steps, a constant or defined above the outermost loop.
      for (mlir::Value idx : extractOp.getIndices()) {
        if (!currOutermostFor.isDefinedOutsideOfLoop(idx) &&
            !(idx.getDefiningOp() &&
              isa<mlir::arith::ConstantOp>(idx.getDefiningOp())) &&
            !isQuasiAffineIVExpressionWithConstantStep(idx)) {
          return mlir::WalkResult::skip();
        }
      }

      // Verify that other args are defined outside the loop nest
      if (!llvm::all_of(scalarOp.getNonBatchableOperands(), [&](mlir::Value v) {
            return currOutermostFor.isDefinedOutsideOfLoop(v);
          })) {
        return mlir::WalkResult::skip();
      }

      // Make sure that there are only loops on the way from the
      // outermost loop to the extract operation (i.e., loops are not
      // embedded in other regions)
      for (Operation *op = extractOp->getParentOp();
           op != currOutermostFor->getParentOp(); op = op->getParentOp()) {
        if (!isa<mlir::scf::ForOp>(op) ||
            !isStaticLoop(llvm::dyn_cast<mlir::scf::ForOp>(op)))
          return mlir::WalkResult::skip();
      }

      targetOp = scalarOp;
      outermostFor = currOutermostFor;
      targetExtractOp = extractOp;

      return mlir::WalkResult::interrupt();
    });

    if (!targetOp)
      return mlir::failure();

    mlir::Value slice = hoistExtractOp(rewriter, outermostFor, targetExtractOp);
    mlir::RankedTensorType sliceType =
        slice.getType().cast<mlir::RankedTensorType>();

    mlir::Value flattenedSlice;
    mlir::ReassociationIndices indices;

    if (sliceType.getRank() == 1) {
      flattenedSlice = slice;
    } else {
      // Flatten the tensor with the batched operands, so that they
      // can be passed as a one-dimensional tensor to the batched
      // operation
      for (int64_t i = 0; i < sliceType.getRank(); i++)
        indices.push_back(i);

      flattenedSlice = rewriter.create<mlir::tensor::CollapseShapeOp>(
          targetExtractOp.getLoc(), slice,
          llvm::SmallVector<mlir::ReassociationIndices>{indices});
    }

    // Create the batched operation and pass flattened, batched
    // operands
    mlir::ImplicitLocOpBuilder ilob(targetExtractOp.getLoc(), rewriter);
    mlir::Value batchedOpResult =
        targetOp.createBatchedOperation(ilob, flattenedSlice);

    mlir::Value expandedBatchResultTensor;

    if (sliceType.getRank() == 1) {
      expandedBatchResultTensor = batchedOpResult;
    } else {
      // Restore original shape of the batched operands for the result
      // of the batched operation. Dimensions, result from indexing
      // with non-loop-IVs are collapsed.
      mlir::Type expandedBatchResultType = mlir::RankedTensorType::get(
          sliceType.getShape(), batchedOpResult.getType()
                                    .dyn_cast<mlir::RankedTensorType>()
                                    .getElementType());

      expandedBatchResultTensor = rewriter.create<mlir::tensor::ExpandShapeOp>(
          targetExtractOp.getLoc(), expandedBatchResultType, batchedOpResult,
          llvm::SmallVector<mlir::ReassociationIndices>{indices});
    }

    // Collect all loop IVs from the extract op. These will be used to
    // index the batched result tensor within the loop for consumers
    // of the batchable op
    llvm::SmallVector<mlir::Value> shiftedLoopIVs;
    ilob.setInsertionPoint(targetOp);

    for (mlir::Value idx : targetExtractOp.getIndices()) {
      mlir::scf::ForOp forOp = findFirstReferencedInductionVar(idx);

      if (forOp) {
        // Loop has either a lower bound that is not 0 or a non-unit
        // step. Shift the index to match the shape of the batched
        // results.
        if (!isStaticNormalizedLoop(forOp)) {
          idx = getOpFoldResultAsValue(
              ilob,
              opFoldDiv(
                  ilob,
                  opFoldSub(ilob,
                            getValueAsOpFoldResult(forOp.getInductionVar()),
                            getValueAsOpFoldResult(forOp.getLowerBound())),
                  getValueAsOpFoldResult(forOp.getStep())));
        }

        shiftedLoopIVs.push_back(idx);
      }
    }

    rewriter.setInsertionPoint(targetOp);
    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(
        targetOp, expandedBatchResultTensor, shiftedLoopIVs);

    return mlir::success();
  }
};

/// Cleanup pattern that replaces a perfect loop nest resulting from
/// repeated application of `BatchingPattern` that only contains a
/// `tensor.extract`, a `tensor.insert` and a `scf.yield` op in the
/// innermost loop, interleaved with side-effect-free operations, with
/// `tensor.extract_slice` and `tensor.insert_slice` ops. E.g.,
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
class CleanupPattern : public mlir::OpRewritePattern<mlir::func::FuncOp> {
public:
  CleanupPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::func::FuncOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp func,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::scf::ForOp outermostFor;
    mlir::tensor::ExtractOp extractOp;
    mlir::tensor::InsertOp insertOp;

    func.walk([&](mlir::tensor::ExtractOp currExtractOp) {
      // First check that the extract op is embedded in a for loop
      mlir::scf::ForOp innermostFor =
          llvm::dyn_cast<mlir::scf::ForOp>(currExtractOp->getParentOp());

      if (!innermostFor)
        return mlir::WalkResult::skip();

      // Next, check find a chain of the 3 relevant operations:
      //
      //  %s  = tensor.extract %T[...]
      //  %U' = tensor.insert %s into %U[...]
      //  scf.yield %U'
      //
      // All other operations must be side-effect-free.
      mlir::Block &body = innermostFor.getRegion().front();

      mlir::scf::YieldOp yield =
          llvm::dyn_cast<mlir::scf::YieldOp>(body.getTerminator());

      if (yield.getOperands().size() != 1)
        return mlir::WalkResult::skip();

      mlir::Operation *yieldOperandProducer =
          yield.getOperand(0).getDefiningOp();

      if (!yieldOperandProducer)
        return mlir::WalkResult::skip();

      mlir::tensor::InsertOp currInsertOp =
          llvm::dyn_cast<mlir::tensor::InsertOp>(yieldOperandProducer);

      if (!currInsertOp ||
          currInsertOp.getScalar() != currExtractOp.getResult())
        return mlir::WalkResult::skip();

      if (!llvm::all_of(body, [&](mlir::Operation &op) {
            return isMemoryEffectFree(&op);
          })) {
        return mlir::WalkResult::skip();
      }

      // Now check that all IVs used for indexation are from perfectly
      // nested loops down to the parent loop of the extract op and
      // identify the outermost loop of the nest
      mlir::scf::ForOp currOutermostFor;

      if (currExtractOp.getIndices().size() !=
          currInsertOp.getIndices().size()) {
        return mlir::WalkResult::skip();
      }

      // Find outermost for loop whose IV is used as an index and make
      // sure that IVs are used for the same indexes of the extract
      // and insert operations
      for (auto it :
           llvm::zip(currExtractOp.getIndices(), currInsertOp.getIndices())) {
        mlir::Value extractIdx = std::get<0>(it);
        mlir::Value insertIdx = std::get<1>(it);

        mlir::scf::ForOp extractForOp;
        mlir::scf::ForOp insertForOp;

        if (!isQuasiAffineIVExpressionWithConstantStep(extractIdx,
                                                       &extractForOp) ||
            !isQuasiAffineIVExpressionWithConstantStep(insertIdx,
                                                       &insertForOp)) {
          return mlir::WalkResult::skip();
        }

        if (insertForOp && extractForOp &&
            insertForOp.getOperation() == extractForOp.getOperation()) {
          if (!currOutermostFor ||
              extractForOp.getOperation()->isAncestor(currOutermostFor)) {
            currOutermostFor = extractForOp;
          }
        }
      }

      if (!currOutermostFor)
        return mlir::WalkResult::skip();

      // Check that the nest from the outermost to the innermost loop
      // is perfect and forwards the result of the innermost loop to
      // the outermost one
      for (mlir::Operation *forOp = innermostFor.getOperation()->getParentOp();
           forOp != currOutermostFor.getOperation()->getParentOp();
           forOp = forOp->getParentOp()) {
        mlir::scf::ForOp currentFor = llvm::dyn_cast<mlir::scf::ForOp>(forOp);

        // Parent is not a for loop
        if (!currentFor)
          return mlir::WalkResult::skip();

        // Body must have exactly two ops: a for loop and a yield
        mlir::Block &body = currentFor.getRegion().front();

        if (body.begin() != std::prev(body.end(), 2))
          return mlir::WalkResult::skip();

        mlir::scf::ForOp childFor =
            llvm::dyn_cast<mlir::scf::ForOp>(*body.begin());
        mlir::scf::YieldOp yield =
            llvm::dyn_cast<mlir::scf::YieldOp>(*std::next(body.begin()));

        // Check that result of the for loop is forwarded
        if (!childFor || !yield || yield.getOperands().size() != 1 ||
            childFor.getResults().size() != 1 ||
            yield.getOperand(0) != childFor.getResult(0))
          return mlir::WalkResult::skip();
      }

      outermostFor = currOutermostFor;
      extractOp = currExtractOp;
      insertOp = currInsertOp;

      return mlir::WalkResult::interrupt();
    });

    if (!outermostFor)
      return mlir::failure();

    // Outermost for loop must produce exactly one result
    if (outermostFor.getInitArgs().size() != 1)
      return mlir::failure();

    // Original tensor that is carried through the loops
    mlir::Value initialTensor = outermostFor.getInitArgs().front();

    mlir::Value slice = hoistExtractOp(rewriter, outermostFor, extractOp);

    mlir::Value insertedSlice =
        hoistInsertOp(rewriter, slice, initialTensor, outermostFor, insertOp);

    // Replace the entire loop nest with the result of the insert
    // slice op. Since this is a perfect loop nest with the innermost
    // body only producing the tensor elements, there cannot be any
    // other operations that produces results or that has side
    // effects.
    rewriter.replaceOp(outermostFor, {insertedSlice});

    return mlir::success();
  }
};

class BatchingPass : public BatchingBase<BatchingPass> {
public:
  void runOnOperation() override {
    mlir::Operation *op = getOperation();

    mlir::RewritePatternSet patterns(op->getContext());
    patterns.add<BatchingPattern, CleanupPattern>(op->getContext());

    if (mlir::applyPatternsAndFoldGreedily(op, std::move(patterns)).failed())
      this->signalPassFailure();
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createBatchingPass() {
  return std::make_unique<BatchingPass>();
}

} // namespace concretelang
} // namespace mlir
