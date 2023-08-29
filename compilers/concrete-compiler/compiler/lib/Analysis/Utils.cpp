#include <concretelang/Analysis/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

using ::concretelang::error::StringError;

namespace mlir {
namespace concretelang {
std::string locationString(mlir::Location loc) {
  auto location = std::string();
  auto locationStream = llvm::raw_string_ostream(location);
  loc->print(locationStream);
  return location;
}

int64_t calculateNumberOfIterations(int64_t start, int64_t stop, int64_t step) {
  int64_t high;
  int64_t low;

  if (step > 0) {
    low = start;
    high = stop;
  } else {
    low = stop;
    high = start;
    step = -step;
  }

  if (low >= high) {
    return 0;
  }

  return ((high - low - 1) / step) + 1;
}

outcome::checked<int64_t, StringError>
calculateNumberOfIterations(scf::ForOp &op) {
  mlir::Value startValue = op.getLowerBound();
  mlir::Value stopValue = op.getUpperBound();
  mlir::Value stepValue = op.getStep();

  auto startOp =
      llvm::dyn_cast_or_null<arith::ConstantOp>(startValue.getDefiningOp());
  auto stopOp =
      llvm::dyn_cast_or_null<arith::ConstantOp>(stopValue.getDefiningOp());
  auto stepOp =
      llvm::dyn_cast_or_null<arith::ConstantOp>(stepValue.getDefiningOp());

  if (!startOp || !stopOp || !stepOp) {
    return StringError("only static loops can be analyzed");
  }

  auto startAttr = startOp.getValue().cast<mlir::IntegerAttr>();
  auto stopAttr = stopOp.getValue().cast<mlir::IntegerAttr>();
  auto stepAttr = stepOp.getValue().cast<mlir::IntegerAttr>();

  if (!startOp || !stopOp || !stepOp) {
    return StringError("only integer loops can be analyzed");
  }

  int64_t start = startAttr.getInt();
  int64_t stop = stopAttr.getInt();
  int64_t step = stepAttr.getInt();

  return calculateNumberOfIterations(start, stop, step);
}
} // namespace concretelang
} // namespace mlir
