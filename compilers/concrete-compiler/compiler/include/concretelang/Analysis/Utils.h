// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_ANALYSIS_UTILS_H
#define CONCRETELANG_ANALYSIS_UTILS_H

#include <boost/outcome.h>
#include <concretelang/Common/Error.h>
#include <limits>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Location.h>

namespace mlir {
namespace concretelang {
class TripCountTracker {
public:
  void pushTripCount(mlir::Operation *op, std::optional<int64_t> n) {
    if (tripCount.has_value()) {
      if (n.has_value()) {
        assert(std::numeric_limits<int64_t>::max() / n.value() >
               tripCount.value());

        tripCount = tripCount.value() * n.value();
      } else {
        savedTripCount = *tripCount;
        tripCount = std::nullopt;
        firstDynamicTripCountOp = op;
      }
    }
  }

  void popTripCount(mlir::Operation *op, std::optional<int64_t> n) {
    if (n.has_value()) {
      if (tripCount.has_value()) {
        tripCount = tripCount.value() / n.value();
      }
    } else {
      if (firstDynamicTripCountOp == op) {
        tripCount = savedTripCount;
      }
      firstDynamicTripCountOp = nullptr;
    }
  }

  std::optional<int64_t> getTripCount() { return tripCount; }

protected:
  std::optional<int64_t> tripCount = 1;
  size_t savedTripCount;
  mlir::Operation *firstDynamicTripCountOp = nullptr;
};

/// Get the string representation of a location
std::string locationString(mlir::Location loc);

} // namespace concretelang
} // namespace mlir

#endif
