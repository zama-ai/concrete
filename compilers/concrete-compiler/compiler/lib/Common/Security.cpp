// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Common/Security.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace concretelang {
namespace security {

double SecurityCurve::getVariance(int glweDimension, int polynomialSize,
                                  int logQ) {
  auto size = glweDimension * polynomialSize;
  if (size < minimalLweDimension) {
    return NAN;
  }
  auto a = std::pow(2, (slope * size + bias) * 2);
  auto b = std::pow(2, -2 * (logQ - 2));
  if (bits == 132) {
    return a + b;
  }
  return a > b ? a : b;
}

#include "concrete/curves.gen.h"

SecurityCurve *getSecurityCurve(int bitsOfSecurity, KeyFormat keyFormat) {
  for (size_t i = 0; i < curvesLen; i++) {
    if (curves[i].bits == bitsOfSecurity && curves[i].keyFormat == keyFormat)
      return &curves[i];
  }
  return nullptr;
}

} // namespace security
} // namespace concretelang
