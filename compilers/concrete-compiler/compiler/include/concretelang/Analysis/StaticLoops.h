// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_ANALYSIS_STATIC_LOOPS_H
#define CONCRETELANG_ANALYSIS_STATIC_LOOPS_H

#include <mlir/Dialect/SCF/IR/SCF.h>

namespace mlir {
namespace concretelang {

/// Convenience class that holds all parameters of a loop
struct LoopsBoundsAndStep {
  int64_t lb;
  int64_t ub;
  int64_t step;

  LoopsBoundsAndStep operator+(const LoopsBoundsAndStep &other) {
    return LoopsBoundsAndStep{lb + other.lb, ub + other.ub, step + other.step};
  }
  LoopsBoundsAndStep operator-(const LoopsBoundsAndStep &other) {
    return LoopsBoundsAndStep{lb - other.lb, ub - other.ub, step - other.step};
  }
  LoopsBoundsAndStep operator*(const LoopsBoundsAndStep &other) {
    return LoopsBoundsAndStep{lb * other.lb, ub * other.ub, step * other.step};
  }
  LoopsBoundsAndStep operator/(int64_t d) {
    return LoopsBoundsAndStep{lb / d, ub / d, step / d};
  }
};

bool isQuasiAffineIVExpression(mlir::Value expr,
                               mlir::scf::ForOp *owningForOp = nullptr);

bool isQuasiAffineIVExpression(mlir::OpFoldResult expr,
                               mlir::scf::ForOp *owningForOp = nullptr);

bool isQuasiAffineIVExpressionWithConstantStep(
    mlir::OpFoldResult expr, mlir::scf::ForOp *forOp = nullptr,
    LoopsBoundsAndStep *basOut = nullptr);

std::optional<LoopsBoundsAndStep>
getBoundsOfQuasiAffineIVExpression(mlir::Value expr, mlir::scf::ForOp forOp);

std::optional<LoopsBoundsAndStep>
getBoundsOfQuasiAffineIVExpression(mlir::OpFoldResult expr,
                                   mlir::scf::ForOp forOp);

int64_t getStaticTripCount(int64_t lb, int64_t ub, int64_t step);
int64_t getStaticTripCount(const LoopsBoundsAndStep &bas);
int64_t getStaticTripCount(mlir::scf::ForOp forOp);
int64_t getNestedStaticTripCount(llvm::ArrayRef<mlir::scf::ForOp> nest);
bool isStaticLoop(mlir::scf::ForOp forOp, int64_t *ilb = nullptr,
                  int64_t *iub = nullptr, int64_t *istep = nullptr);

bool isConstantIndexValue(mlir::Value v);
int64_t getConstantIndexValue(mlir::Value v);

} // namespace concretelang
} // namespace mlir

#endif
