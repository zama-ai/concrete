// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_GLOBALFHECONTEXT_H_
#define CONCRETELANG_CONVERSION_GLOBALFHECONTEXT_H_
#include <cstddef>
#include <cstdint>
#include <vector>

#include "concretelang/Support/V0Parameters.h"
#include "llvm/ADT/Optional.h"

namespace mlir {
namespace concretelang {

struct V0FHEContext {
  V0FHEContext() = delete;
  V0FHEContext(const V0FHEConstraint &constraint,
               const optimizer::Solution solution)
      : constraint(constraint), solution(solution) {}

  V0FHEConstraint constraint;
  optimizer::Solution solution;
};
} // namespace concretelang
} // namespace mlir

#endif
