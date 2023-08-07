#include <concretelang/Dialect/Concrete/Analysis/MemoryUsage.h>
#include <concretelang/Dialect/Concrete/IR/ConcreteOps.h>
#include <concretelang/Support/logging.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <numeric>

using namespace mlir::concretelang;
using namespace mlir;

using Concrete::MemoryUsagePass;

namespace {

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

std::optional<StringError> calculateNumberOfIterations(scf::ForOp &op,
                                                       int64_t &result) {
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

  result = calculateNumberOfIterations(start, stop, step);
  return std::nullopt;
}

static std::optional<StringError> on_enter(scf::ForOp &op,
                                           MemoryUsagePass &pass) {
  int64_t numberOfIterations;

  std::optional<StringError> error =
      calculateNumberOfIterations(op, numberOfIterations);
  if (error.has_value()) {
    return error;
  }

  assert(numberOfIterations > 0);
  pass.iterations *= (uint64_t)numberOfIterations;
  return std::nullopt;
}

static std::optional<StringError> on_exit(scf::ForOp &op,
                                          MemoryUsagePass &pass) {
  int64_t numberOfIterations;

  std::optional<StringError> error =
      calculateNumberOfIterations(op, numberOfIterations);
  if (error.has_value()) {
    return error;
  }

  assert(numberOfIterations > 0);
  pass.iterations /= (uint64_t)numberOfIterations;
  return std::nullopt;
}

int64_t getElementTypeSize(mlir::Type elementType) {
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    auto width = integerType.getWidth();
    return std::ceil((double)width / 8);
  }
  if (mlir::dyn_cast<mlir::IndexType>(elementType)) {
    return 8;
  }
  return -1;
}

outcome::checked<int64_t, StringError>
getBufferSize(mlir::MemRefType bufferType) {
  auto shape = bufferType.getShape();
  auto elementType = bufferType.getElementType();
  auto elementSize = getElementTypeSize(elementType);
  if (elementSize == -1)
    return StringError(
        "allocation of buffer with a non-supported element-type");
  for (auto size : shape) {
    if (size == mlir::ShapedType::kDynamic) {
      log_verbose() << "warning: dynamic dimension found during computation of "
                       "memory usage. Dynamic size will be ignored";
    }
  }
  auto multiply_ignore_dyn_size = [](int64_t size_1, int64_t size_2) {
    // we don't want to multiply by a dynamic size
    return size_2 == mlir::ShapedType::kDynamic ? size_1 : size_1 * size_2;
  };

  return elementSize * std::accumulate(shape.begin(), shape.end(), 1,
                                       multiply_ignore_dyn_size);
}

bool isBufferDeallocated(mlir::Value buffer) {
  for (auto user : buffer.getUsers()) {
    if (mlir::isa<memref::DeallocOp>(user))
      return true;
  }
  return false;
}

static std::optional<StringError> on_enter(memref::AllocOp &op,
                                           MemoryUsagePass &pass) {

  auto maybeBufferSize = getBufferSize(op.getResult().getType());
  if (!maybeBufferSize) {
    return maybeBufferSize.error();
  }
  // if the allocated buffer is being deallocated then count it as one.
  // Otherwise (and there must be a problem) multiply it by the number of
  // iterations
  int64_t numberOfAlloc =
      isBufferDeallocated(op.getResult()) ? 1 : pass.iterations;

  auto location = locationString(op.getLoc());
  // pass.iterations number of allocation of size: shape_1 * ... * shape_n *
  // element_size
  auto memoryUsage = numberOfAlloc * maybeBufferSize.value();

  pass.feedback.memoryUsagePerLoc[location] += memoryUsage;

  return std::nullopt;
}

static std::optional<StringError> on_enter(mlir::Operation *op,
                                           MemoryUsagePass &pass) {
  for (auto operand : op->getOperands()) {
    // we only consider buffers
    if (!mlir::isa<mlir::MemRefType>(operand.getType()))
      continue;
    // find the origin of the buffer
    auto definingOp = operand.getDefiningOp();
    mlir::Value lastVisitedBuffer = operand;
    while (definingOp) {
      mlir::ViewLikeOpInterface viewLikeOp =
          mlir::dyn_cast<mlir::ViewLikeOpInterface>(definingOp);
      if (viewLikeOp) {
        lastVisitedBuffer = viewLikeOp.getViewSource();
        definingOp = lastVisitedBuffer.getDefiningOp();
      } else {
        break;
      }
    }
    // we already count allocations separately
    if (definingOp && mlir::isa<memref::AllocOp>(definingOp) &&
        definingOp->getLoc() == op->getLoc())
      continue;

    auto location = locationString(op->getLoc());

    std::vector<mlir::Value> &visited = pass.visitedValuesPerLoc[location];

    // the search would be faster if we use an unsorted_set, but we need a hash
    // function for mlir::Value
    if (std::find(visited.begin(), visited.end(), lastVisitedBuffer) ==
        visited.end()) {
      visited.push_back(lastVisitedBuffer);

      auto maybeBufferSize =
          getBufferSize(lastVisitedBuffer.getType().cast<mlir::MemRefType>());
      if (!maybeBufferSize) {
        return maybeBufferSize.error();
      }
      auto bufferSize = maybeBufferSize.value();

      pass.feedback.memoryUsagePerLoc[location] += bufferSize;
    }
  }

  return std::nullopt;
}

} // namespace

std::optional<StringError> MemoryUsagePass::enter(mlir::Operation *op) {
  // specialized calls
  if (auto typedOp = llvm::dyn_cast<scf::ForOp>(op)) {
    std::optional<StringError> error = on_enter(typedOp, *this);
    if (error.has_value()) {
      return error;
    }
  }
  if (auto typedOp = llvm::dyn_cast<memref::AllocOp>(op)) {
    std::optional<StringError> error = on_enter(typedOp, *this);
    if (error.has_value()) {
      return error;
    }
  }

  // call generic enter
  std::optional<StringError> error = on_enter(op, *this);
  if (error.has_value()) {
    return error;
  }
  return std::nullopt;
}

std::optional<StringError> MemoryUsagePass::exit(mlir::Operation *op) {
  if (auto typedOp = llvm::dyn_cast<scf::ForOp>(op)) {
    std::optional<StringError> error = on_exit(typedOp, *this);
    if (error.has_value()) {
      return error;
    }
  }
  return std::nullopt;
}
