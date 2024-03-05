// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_TFHEKEYNORMALIZATION_PASS_H_
#define CONCRETELANG_CONVERSION_TFHEKEYNORMALIZATION_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace concretelang {
/// Create a pass that ensures that the ids of the keys form a proper index
/// range.
std::unique_ptr<OperationPass<ModuleOp>> createTFHEKeyNormalizationPass();
} // namespace concretelang
} // namespace mlir

#endif
