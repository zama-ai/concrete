
#ifndef ZAMALANG_CONVERSION_LOWLFHEUNPARAMETRIZE_PASS_H_
#define ZAMALANG_CONVERSION_LOWLFHEUNPARAMETRIZE_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace zamalang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLowLFHEUnparametrizePass();
} // namespace zamalang
} // namespace mlir

#endif