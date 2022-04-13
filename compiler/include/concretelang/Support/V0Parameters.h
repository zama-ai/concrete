// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_V0Parameter_H_
#define CONCRETELANG_SUPPORT_V0Parameter_H_

#include "llvm/ADT/Optional.h"

#include "concretelang/Conversion/Utils/GlobalFHEContext.h"

namespace mlir {
namespace concretelang {

llvm::Optional<V0Parameter> getV0Parameter(V0FHEConstraint constraint);

} // namespace concretelang
} // namespace mlir
#endif
