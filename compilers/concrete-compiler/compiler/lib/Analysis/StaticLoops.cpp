// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <concretelang/Analysis/StaticLoops.h>

namespace mlir {
namespace concretelang {

/// Checks whether the expression `expr` is a quasi-affine expression
/// on a single induction variable. If an induction variable is
/// referenced, the owning for loop is returned in `*owningForOp`.
bool isQuasiAffineIVExpression(mlir::Value expr,
                               mlir::scf::ForOp *owningForOp) {
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
    } else if (mlir::AffineApplyOp applyOp =
                   llvm::dyn_cast<mlir::AffineApplyOp>(op)) {
      // Affine.apply: make sure that all operands are either constant
      // expressions or using IVs of the same loop
      mlir::scf::ForOp ivOwner;

      for (mlir::Value operand : applyOp->getOperands()) {
        mlir::scf::ForOp thisOwner;
        if (!isQuasiAffineIVExpression(operand, &thisOwner))
          return false;

        if (thisOwner) {
          if (!ivOwner) {
            ivOwner = thisOwner;
          } else {
            if (thisOwner != ivOwner)
              return false;
          }
        }
      }

      if (owningForOp != nullptr)
        *owningForOp = ivOwner;
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

bool isQuasiAffineIVExpression(mlir::OpFoldResult expr,
                               mlir::scf::ForOp *owningForOp) {
  if (mlir::Value dynExpr = expr.dyn_cast<mlir::Value>())
    return isQuasiAffineIVExpression(dynExpr, owningForOp);

  return true;
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
bool isQuasiAffineIVExpressionWithConstantStep(mlir::OpFoldResult expr,
                                               mlir::scf::ForOp *forOp,
                                               LoopsBoundsAndStep *basOut) {
  mlir::scf::ForOp tmpForOp;

  if (isQuasiAffineIVExpression(expr, &tmpForOp)) {
    std::optional<LoopsBoundsAndStep> bas =
        getBoundsOfQuasiAffineIVExpression(expr, tmpForOp);

    if (bas.has_value()) {
      if (forOp != nullptr)
        *forOp = tmpForOp;
      if (basOut != nullptr)
        *basOut = *bas;

      return true;
    }
  }

  return false;
}

static std::optional<LoopsBoundsAndStep>
getBoundsOfAffineExpression(mlir::AffineExpr expr,
                            llvm::ArrayRef<LoopsBoundsAndStep> dimBounds);

// Returns the static bounds of an affine binary expression `expr`
// given the bounds `dimBounds` for any dimension expression appearing
// in the affine expression by determining the bounds for the left
// hand side and right hand side separately and applying `combinator`
// on them.
static std::optional<LoopsBoundsAndStep> getBoundsOfAffineBinaryExpression(
    mlir::AffineExpr expr, llvm::ArrayRef<LoopsBoundsAndStep> dimBounds,
    llvm::function_ref<LoopsBoundsAndStep(LoopsBoundsAndStep,
                                          LoopsBoundsAndStep)>
        combinator) {
  mlir::AffineBinaryOpExpr binExpr = expr.cast<mlir::AffineBinaryOpExpr>();

  std::optional<LoopsBoundsAndStep> lhs =
      getBoundsOfAffineExpression(binExpr.getLHS(), dimBounds);

  if (!lhs.has_value())
    return std::nullopt;

  std::optional<LoopsBoundsAndStep> rhs =
      getBoundsOfAffineExpression(binExpr.getRHS(), dimBounds);

  if (!rhs.has_value())
    return std::nullopt;

  return combinator(lhs.value(), rhs.value());
}

// Returns the static bounds of an affine expression given the bounds
// `dimBounds` for any dimension expression appearing in the affine
// expression.
static std::optional<LoopsBoundsAndStep>
getBoundsOfAffineExpression(mlir::AffineExpr expr,
                            llvm::ArrayRef<LoopsBoundsAndStep> dimBounds) {
  // Cannot just use AffineExpr::compose() due to the check on
  // division
  switch (expr.getKind()) {
  case mlir::AffineExprKind::SymbolId:
    assert(false &&
           "Symbol found in affine expression that should not contain sumbols");
    break;
  case mlir::AffineExprKind::Constant: {
    int64_t cstVal = expr.cast<mlir::AffineConstantExpr>().getValue();
    return LoopsBoundsAndStep{cstVal, cstVal, 0};
  }
  case mlir::AffineExprKind::DimId: {
    unsigned dimId = expr.cast<AffineDimExpr>().getPosition();
    assert(dimId < dimBounds.size());
    return dimBounds[dimId];
  }
  case AffineExprKind::Add:
    return getBoundsOfAffineBinaryExpression(
        expr, dimBounds, [](LoopsBoundsAndStep lhs, LoopsBoundsAndStep rhs) {
          return lhs + rhs;
        });
  case AffineExprKind::Mul:
    return getBoundsOfAffineBinaryExpression(
        expr, dimBounds, [](LoopsBoundsAndStep lhs, LoopsBoundsAndStep rhs) {
          return lhs * rhs;
        });

  case AffineExprKind::Mod:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::FloorDiv: {
    mlir::AffineBinaryOpExpr binExpr = expr.cast<mlir::AffineBinaryOpExpr>();

    std::optional<LoopsBoundsAndStep> lhs =
        getBoundsOfAffineExpression(binExpr.getLHS(), dimBounds);
    std::optional<LoopsBoundsAndStep> rhs =
        getBoundsOfAffineExpression(binExpr.getRHS(), dimBounds);

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

  llvm_unreachable("Unknown affine expression kind");
}

// Returns the static bounds for the affine map `map` given the static
// bounds for all operands on which the map is applied. The map must
// not contain any symbols, all of its expressions must be pure affine
// expressions and the number of results must be one.
static std::optional<LoopsBoundsAndStep>
getBoundsOfAffineMap(mlir::AffineMap map,
                     llvm::ArrayRef<LoopsBoundsAndStep> mapOperandBounds) {
  assert(map.getNumResults() == 1 &&
         "Attempting to get bounds for map with multiple result dimensions");
  assert(map.getNumSymbols() == 0 &&
         "Attempting to get bounds for map with symbols");
  assert(map.getResult(0).isPureAffine() &&
         "Attempting to get bounds for non-pure affine expression");

  return getBoundsOfAffineExpression(map.getResult(0), mapOperandBounds);
}

/// Returns the lower bound, upper bound and step of the quasi-affine
/// expression `expr` on the the induction variable from a for
/// operation.
std::optional<LoopsBoundsAndStep>
getBoundsOfQuasiAffineIVExpression(mlir::Value expr, mlir::scf::ForOp forOp) {
  // Base case: expression is the induction variable itself -> check
  // if the bounds are static and return them
  if (forOp && expr == forOp.getInductionVar() &&
      isConstantIndexValue(forOp.getLowerBound()) &&
      isConstantIndexValue(forOp.getUpperBound()) &&
      isConstantIndexValue(forOp.getStep())) {
    return LoopsBoundsAndStep{getConstantIndexValue(forOp.getLowerBound()),
                              getConstantIndexValue(forOp.getUpperBound()),
                              getConstantIndexValue(forOp.getStep())};
  }
  // Arithmetic expression
  else if (mlir::Operation *op = expr.getDefiningOp()) {
    if (llvm::isa<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp,
                  mlir::arith::DivSIOp>(op)) {

      std::optional<LoopsBoundsAndStep> lhs =
          getBoundsOfQuasiAffineIVExpression(op->getOperand(0), forOp);
      std::optional<LoopsBoundsAndStep> rhs =
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
    // Affine.apply
    if (mlir::AffineApplyOp applyOp = llvm::dyn_cast<mlir::AffineApplyOp>(op)) {
      if (applyOp.getMap().getNumResults() != 1 ||
          applyOp.getMap().getNumSymbols() != 0 ||
          !applyOp.getMap().getResult(0).isPureAffine())
        return std::nullopt;

      llvm::SmallVector<LoopsBoundsAndStep> bounds;

      for (mlir::Value operand : applyOp.getMapOperands()) {
        std::optional<LoopsBoundsAndStep> operatorBounds =
            getBoundsOfQuasiAffineIVExpression(operand, forOp);

        if (!operatorBounds.has_value())
          return std::nullopt;

        bounds.push_back(operatorBounds.value());
      }

      return getBoundsOfAffineMap(applyOp.getMap(), bounds);
    }
    // Base case: constant -> return constant value
    else if (llvm::isa<mlir::arith::ConstantIndexOp>(expr.getDefiningOp())) {
      mlir::arith::ConstantIndexOp cst =
          llvm::dyn_cast<mlir::arith::ConstantIndexOp>(expr.getDefiningOp());
      return LoopsBoundsAndStep{cst.value(), cst.value(), 0};
    }
  }

  return std::nullopt;
}

std::optional<LoopsBoundsAndStep>
getBoundsOfQuasiAffineIVExpression(mlir::OpFoldResult expr,
                                   mlir::scf::ForOp forOp) {
  if (mlir::Value dynExpr = expr.dyn_cast<mlir::Value>())
    return getBoundsOfQuasiAffineIVExpression(dynExpr, forOp);

  mlir::IntegerAttr exprAttr =
      expr.dyn_cast<mlir::Attribute>().dyn_cast_or_null<mlir::IntegerAttr>();

  assert(exprAttr && "Expected OpFoldResult to contain either a Value or an "
                     "integer attribute");

  return LoopsBoundsAndStep{exprAttr.getInt(), exprAttr.getInt(), 0};
}

/// Checks if `forOp` has constant bounds and a constant step
/// resulting from quasi affine expressions.
bool isStaticLoop(mlir::scf::ForOp forOp, int64_t *ilb, int64_t *iub,
                  int64_t *istep) {
  std::optional<LoopsBoundsAndStep> basLB =
      getBoundsOfQuasiAffineIVExpression(forOp.getLowerBound(), nullptr);
  std::optional<LoopsBoundsAndStep> basUB =
      getBoundsOfQuasiAffineIVExpression(forOp.getUpperBound(), nullptr);
  std::optional<LoopsBoundsAndStep> basStep =
      getBoundsOfQuasiAffineIVExpression(forOp.getStep(), nullptr);

  if (!basLB.has_value() || !basUB.has_value() || !basStep.has_value())
    return false;

  if ((basLB->lb != basLB->ub || basLB->step != 0) ||
      (basUB->lb != basUB->ub || basUB->step != 0) ||
      (basStep->lb != basStep->ub || basStep->step != 0))
    return false;

  if (ilb)
    *ilb = basLB->lb;

  if (iub)
    *iub = basUB->lb;

  if (istep)
    *istep = basStep->lb;

  return true;
}

int64_t getStaticTripCount(int64_t lb, int64_t ub, int64_t step) {
  assert((step == 0 && lb == ub) || (step >= 0 && lb <= ub) ||
         (step < 0 && lb > ub));

  if (lb == ub)
    return 0;

  if (lb > ub)
    return getStaticTripCount(ub, lb, -step);

  assert(ub - lb < std::numeric_limits<int64_t>::max() - step);

  return (ub - lb + step - 1) / step;
}

int64_t getStaticTripCount(const LoopsBoundsAndStep &bas) {
  return getStaticTripCount(bas.lb, bas.ub, bas.step);
}

// Returns the number of iterations of a static loop
int64_t getStaticTripCount(mlir::scf::ForOp forOp) {
  int64_t lb;
  int64_t ub;
  int64_t step;

  bool isStatic = isStaticLoop(forOp, &lb, &ub, &step);

  assert(isStatic && "Loop must be static");

  return getStaticTripCount(lb, ub, step);
}

// Returns the total number of executions of the body of the innermost
// loop of a nest of static loops
int64_t getNestedStaticTripCount(llvm::ArrayRef<mlir::scf::ForOp> nest) {
  int64_t tripCount = 1;

  for (mlir::scf::ForOp forOp : nest) {
    int64_t thisCount = getStaticTripCount(forOp);

    if (thisCount == 0)
      return 0;

    assert(std::numeric_limits<int64_t>::max() / thisCount >= tripCount);
    tripCount *= thisCount;
  }

  return tripCount;
}

// Checks whether `v` is a constant value of type index
bool isConstantIndexValue(mlir::Value v) {
  return v.getDefiningOp() &&
         llvm::isa<mlir::arith::ConstantIndexOp>(*v.getDefiningOp());
}

/// Assumes that `v` is a constant index operation and returns the
/// constant value as an `int64_t`.
int64_t getConstantIndexValue(mlir::Value v) {
  assert(isConstantIndexValue(v));

  return llvm::dyn_cast<mlir::arith::ConstantIndexOp>(*v.getDefiningOp())
      .value();
}

} // namespace concretelang
} // namespace mlir
