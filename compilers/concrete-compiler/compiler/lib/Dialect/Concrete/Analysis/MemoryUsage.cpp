#include <concretelang/Analysis/StaticLoops.h>
#include <concretelang/Analysis/Utils.h>
#include <concretelang/Dialect/Concrete/Analysis/MemoryUsage.h>
#include <concretelang/Dialect/Concrete/IR/ConcreteOps.h>
#include <concretelang/Dialect/RT/IR/RTTypes.h>
#include <concretelang/Support/logging.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <numeric>
#include <optional>

using namespace mlir::concretelang;
using namespace mlir;

namespace {
outcome::checked<int64_t, StringError>
getBufferSize(mlir::MemRefType bufferType);

outcome::checked<int64_t, StringError>
getElementTypeSize(mlir::Type elementType) {
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    auto width = integerType.getWidth();
    return std::ceil((double)width / 8);
  } else if (mlir::dyn_cast<mlir::IndexType>(elementType)) {
    return 8;
  } else if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(elementType)) {
    return getBufferSize(memrefType);
  } else if (auto futureType = mlir::dyn_cast<RT::FutureType>(elementType)) {
    auto recSize = getElementTypeSize(futureType.getElementType());

    if (!recSize)
      return recSize.error();

    // FIXME: Hardcoded size of 8 bytes for a pointer
    return 8 * recSize.value();
  }

  return -1;
}

outcome::checked<int64_t, StringError>
getBufferSize(mlir::MemRefType bufferType) {
  auto shape = bufferType.getShape();
  auto elementType = bufferType.getElementType();
  auto maybeElementSize = getElementTypeSize(elementType);

  if (!maybeElementSize)
    return maybeElementSize.error();

  auto elementSize = maybeElementSize.value();

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

} // namespace

namespace mlir {
namespace concretelang {
namespace Concrete {

namespace {
// Adds the values of `a` and `b` if they are both defined,
// otherwise returns `std::nullopt`
std::optional<int64_t> addOrNullopt(std::optional<int64_t> a,
                                    std::optional<int64_t> b) {
  if (a.has_value() && b.has_value())
    return a.value() + b.value();
  else
    return std::nullopt;
}
} // namespace

struct MemoryUsagePass
    : public PassWrapper<MemoryUsagePass, OperationPass<ModuleOp>>,
      public TripCountTracker {

  ProgramCompilationFeedback &feedback;
  CircuitCompilationFeedback *circuitFeedback;

  MemoryUsagePass(ProgramCompilationFeedback &feedback)
      : feedback{feedback}, circuitFeedback{nullptr} {};

  void runOnOperation() override {
    auto module = getOperation();
    auto funcs = module.getOps<mlir::func::FuncOp>();
    for (CircuitCompilationFeedback &circuitFeedback :
         feedback.circuitFeedbacks) {
      auto funcOp = llvm::find_if(funcs, [&](mlir::func::FuncOp op) {
        return op.getName() == circuitFeedback.name;
      });
      assert(funcOp != funcs.end());
      this->circuitFeedback = &circuitFeedback;

      WalkResult walk =
          getOperation()->walk([&](Operation *op, const WalkStage &stage) {
            if (stage.isBeforeAllRegions()) {
              std::optional<StringError> error = this->enter(op);
              if (error.has_value()) {
                op->emitError() << error->mesg;
                return WalkResult::interrupt();
              }
            }

            if (stage.isAfterAllRegions()) {
              std::optional<StringError> error = this->exit(op);
              if (error.has_value()) {
                op->emitError() << error->mesg;
                return WalkResult::interrupt();
              }
            }

            return WalkResult::advance();
          });

      if (walk.wasInterrupted()) {
        signalPassFailure();
        return;
      }
    }
  }

  std::optional<StringError> enter(mlir::Operation *op) {
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

  std::optional<StringError> exit(mlir::Operation *op) {
    if (auto typedOp = llvm::dyn_cast<scf::ForOp>(op)) {
      std::optional<StringError> error = on_exit(typedOp, *this);
      if (error.has_value()) {
        return error;
      }
    }
    return std::nullopt;
  }

  static std::optional<StringError> on_enter(scf::ForOp &op,
                                             MemoryUsagePass &pass) {
    std::optional<int64_t> tripCount = tryGetStaticTripCount(op);

    if (!tripCount.has_value()) {
      emitWarning(op.getLoc(), "Cannot determine static trip count");
    }

    pass.pushTripCount(op, tripCount);

    return std::nullopt;
  }

  static std::optional<StringError> on_exit(scf::ForOp &op,
                                            MemoryUsagePass &pass) {
    std::optional<int64_t> tripCount = tryGetStaticTripCount(op);
    pass.popTripCount(op, tripCount);

    return std::nullopt;
  }

  static std::optional<StringError> on_enter(memref::AllocOp &op,
                                             MemoryUsagePass &pass) {

    auto maybeBufferSize = getBufferSize(op.getResult().getType());
    if (!maybeBufferSize) {
      return maybeBufferSize.error();
    }

    std::optional<int64_t> memoryUsage = maybeBufferSize.value();

    // if the allocated buffer is being deallocated then count it as one.
    // Otherwise (and there must be a problem) multiply it by the number of
    // iterations
    if (!isBufferDeallocated(op.getResult())) {
      if (pass.getTripCount().has_value())
        memoryUsage = memoryUsage.value() * pass.getTripCount().value();
      else
        memoryUsage = std::nullopt;
    }

    auto location = locationString(op.getLoc());

    if (pass.circuitFeedback->memoryUsagePerLoc.find(location) !=
        pass.circuitFeedback->memoryUsagePerLoc.end()) {
      pass.circuitFeedback->memoryUsagePerLoc[location] = addOrNullopt(
          pass.circuitFeedback->memoryUsagePerLoc[location], memoryUsage);
    } else {
      pass.circuitFeedback->memoryUsagePerLoc[location] = memoryUsage;
    }

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

      // the search would be faster if we use an unsorted_set, but we need a
      // hash function for mlir::Value
      if (std::find(visited.begin(), visited.end(), lastVisitedBuffer) ==
          visited.end()) {
        visited.push_back(lastVisitedBuffer);

        auto maybeBufferSize =
            getBufferSize(lastVisitedBuffer.getType().cast<mlir::MemRefType>());
        if (!maybeBufferSize) {
          return maybeBufferSize.error();
        }
        auto bufferSize = maybeBufferSize.value();

        if (pass.circuitFeedback->memoryUsagePerLoc.find(location) !=
            pass.circuitFeedback->memoryUsagePerLoc.end()) {
          pass.circuitFeedback->memoryUsagePerLoc[location] = addOrNullopt(
              pass.circuitFeedback->memoryUsagePerLoc[location], bufferSize);
        } else {
          pass.circuitFeedback->memoryUsagePerLoc[location] = bufferSize;
        }
      }
    }

    return std::nullopt;
  }

  std::map<std::string, std::vector<mlir::Value>> visitedValuesPerLoc;
};

} // namespace Concrete

std::unique_ptr<OperationPass<ModuleOp>>
createMemoryUsagePass(ProgramCompilationFeedback &feedback) {
  return std::make_unique<Concrete::MemoryUsagePass>(feedback);
}

} // namespace concretelang
} // namespace mlir
