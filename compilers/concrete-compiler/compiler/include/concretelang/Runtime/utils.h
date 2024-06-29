// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_UTILS_H
#define CONCRETELANG_RUNTIME_UTILS_H

#include "llvm/Support/TargetSelect.h"

namespace mlir {
namespace concretelang {

// Mainly a wrapper to some LLVM functions. The reason to have this wrapper is
// to avoid linking conflicts between the python binary extension, and LLVM.
void LLVMInitializeNativeTarget();

} // namespace concretelang
} // namespace mlir

#endif
