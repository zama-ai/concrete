// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/time_util.h"

#if CONCRETELANG_TIMING_ENABLED

namespace mlir {
namespace concretelang {
namespace time_util {

bool timing_enabled = false;
struct timespec timestamp;

} // namespace time_util
} // namespace concretelang
} // namespace mlir

#endif
