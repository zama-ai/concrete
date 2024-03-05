// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_MLIRLOWERABLEDIALECTSTOLLVM_PASS_H_
#define CONCRETELANG_CONVERSION_MLIRLOWERABLEDIALECTSTOLLVM_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
template <typename T> class OperationPass;
namespace concretelang {
/// Create a pass to convert MLIR lowerable dialects to LLVM.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertMLIRLowerableDialectsToLLVMPass();
} // namespace concretelang
} // namespace mlir

#endif
