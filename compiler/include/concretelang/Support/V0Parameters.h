// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_V0Parameter_H_
#define CONCRETELANG_SUPPORT_V0Parameter_H_

#include "llvm/ADT/Optional.h"

#include "concretelang/Conversion/Utils/GlobalFHEContext.h"

namespace mlir {
namespace concretelang {

namespace optimizer {
constexpr double P_ERROR_4_SIGMA = 1.0 - 0.999936657516;
struct Config {
  double p_error;
  bool display;
};
constexpr Config DEFAULT_CONFIG = {P_ERROR_4_SIGMA, false};
} // namespace optimizer

llvm::Optional<V0Parameter> getV0Parameter(V0FHEConstraint constraint,
                                           optimizer::Config optimizerConfig);

} // namespace concretelang
} // namespace mlir
#endif
