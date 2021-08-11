
#ifndef ZAMALANG_CONVERSION_MIDLFHETOLOWLFHE_PASS_H_
#define ZAMALANG_CONVERSION_MIDLFHETOLOWLFHE_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace zamalang {
/// Create a pass to convert `MidLFHE` dialect to `LowLFHE` dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertMidLFHEToLowLFHEPass();
} // namespace zamalang
} // namespace mlir

#endif