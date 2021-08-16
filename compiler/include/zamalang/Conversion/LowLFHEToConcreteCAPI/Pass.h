
#ifndef ZAMALANG_CONVERSION_LOWLFHETOCONCRETECAPI_PASS_H_
#define ZAMALANG_CONVERSION_LOWLFHETOCONCRETECAPI_PASS_H_

#include "mlir/Pass/Pass.h"

#include "zamalang/Conversion/Utils/GlobalFHEContext.h"

namespace mlir {
namespace zamalang {
/// Create a pass to convert `LowLFHE` operators to function call to the
/// `ConcreteCAPI`
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLowLFHEToConcreteCAPIPass();
} // namespace zamalang
} // namespace mlir

#endif