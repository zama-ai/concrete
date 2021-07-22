
#ifndef ZAMALANG_CONVERSION_HLFHETOMIDLFHE_PASS_H_
#define ZAMALANG_CONVERSION_HLFHETOMIDLFHE_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace zamalang {
/// Create a pass to convert `HLFHE` dialect to `MidLFHE` dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertHLFHEToMidLFHEPass();
} // namespace zamalang
} // namespace mlir

#endif