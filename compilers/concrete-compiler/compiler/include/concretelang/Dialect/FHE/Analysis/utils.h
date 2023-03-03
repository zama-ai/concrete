// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_FHE_ANALYSIS_UTILS_H
#define CONCRETELANG_DIALECT_FHE_ANALYSIS_UTILS_H

#include <mlir/IR/BuiltinOps.h>

namespace mlir {
namespace concretelang {
namespace fhe {
namespace utils {

bool isEncryptedValue(mlir::Value value);
unsigned int getEintPrecision(mlir::Value value);

} // namespace utils
} // namespace fhe
} // namespace concretelang
} // namespace mlir

#endif
