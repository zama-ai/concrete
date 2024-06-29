// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/utils.h"

namespace mlir {
namespace concretelang {
void LLVMInitializeNativeTarget() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
}
} // namespace concretelang
} // namespace mlir
