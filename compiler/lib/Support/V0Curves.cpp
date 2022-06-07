// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cmath>

#include "concretelang/Support/V0Curves.h"

namespace mlir {
namespace concretelang {

V0Curves curves[SECURITY_LEVEL_MAX][KEY_FORMAT_MAX] = {
    {V0Curves(SECURITY_LEVEL_80, -0.04047677865612648, 1.1433465085639063, 160,
              1)},
    {V0Curves(SECURITY_LEVEL_128, -0.026374888765705498, 2.012143923330495, 256,
              1)},
    {V0Curves(SECURITY_LEVEL_192, -0.018504919354426233, 2.6634073426215843,
              381, 1)},
    {V0Curves(SECURITY_LEVEL_256, -0.014327640360322604, 2.899270827311091, 781,
              1)}};

V0Curves *getV0Curves(int securityLevel, int keyFormat) {
  if (securityLevel >= SECURITY_LEVEL_MAX || keyFormat >= KEY_FORMAT_MAX) {
    return nullptr;
  }
  return &curves[securityLevel][keyFormat];
}
} // namespace concretelang
} // namespace mlir
