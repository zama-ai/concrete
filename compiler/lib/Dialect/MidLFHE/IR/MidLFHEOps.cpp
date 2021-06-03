#include "mlir/IR/Region.h"

#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

namespace mlir {
namespace zamalang {
bool predPBSRegion(::mlir::Region &region) {
  if (region.getBlocks().size() != 1) {
    return false;
  }
  auto args = region.getBlocks().front().getArguments();
  if (args.size() != 1) {
    return false;
  }
  return args.front().getType().isa<mlir::IntegerType>();
}
} // namespace zamalang
} // namespace mlir

#define GET_OP_CLASSES
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.cpp.inc"
