// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cmath>

#include "concretelang/Support/V0Curves.h"

namespace mlir {
namespace concretelang {

V0Curves curves[SECURITY_LEVEL_MAX][KEY_FORMAT_MAX] = {
    {V0Curves(SECURITY_LEVEL_80, -0.04042633119364589, 1.6609788641436722, 450,
              1)},
    {V0Curves(SECURITY_LEVEL_128, -0.02640502876522622, 2.4826422691043177, 450,
              1)},
    {V0Curves(SECURITY_LEVEL_192, -0.018610403247590085, 3.2996236848399008,
              606, 1)},
    {V0Curves(SECURITY_LEVEL_256, -0.014606812351714953, 3.8493629234693003,
              826, 1)}};

V0Curves *getV0Curves(int securityLevel, int keyFormat) {
  if (securityLevel >= SECURITY_LEVEL_MAX || keyFormat >= KEY_FORMAT_MAX) {
    return nullptr;
  }
  return &curves[securityLevel][keyFormat];
}
} // namespace concretelang
} // namespace mlir
