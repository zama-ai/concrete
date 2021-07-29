
#ifndef ZAMALANG_CONVERSION_MLIRLOWERABLEDIALECTSTOLLVM_PASS_H_
#define ZAMALANG_CONVERSION_MLIRLOWERABLEDIALECTSTOLLVM_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
template <typename T> class OperationPass;
namespace zamalang {
/// Create a pass to convert MLIR lowerable dialects to LLVM.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertMLIRLowerableDialectsToLLVMPass();
} // namespace zamalang
} // namespace mlir

#endif