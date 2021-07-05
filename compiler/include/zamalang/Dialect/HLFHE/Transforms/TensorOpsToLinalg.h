#ifndef ZAMALANG_DIALECT_HLFHE_TRANSFORMS_TENSOROPSTOLINALG_H
#define ZAMALANG_DIALECT_HLFHE_TRANSFORMS_TENSOROPSTOLINALG_H

#include <mlir/Pass/Pass.h>

namespace mlir {
namespace zamalang {
namespace HLFHE {
std::unique_ptr<mlir::Pass> createLowerTensorOpsToLinalgPass();
} // namespace HLFHE
} // namespace zamalang
} // namespace mlir

#endif
