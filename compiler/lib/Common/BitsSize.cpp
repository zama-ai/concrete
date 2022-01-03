// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "concretelang/Common/BitsSize.h"

namespace concretelang {
namespace common {

size_t bitWidthAsWord(size_t exactBitWidth) {
  size_t sortedWordBitWidths[] = {8, 16, 32, 64};
  size_t previousWidth = 0;
  for (auto currentWidth : sortedWordBitWidths) {
    if (previousWidth < exactBitWidth && exactBitWidth <= currentWidth) {
      return currentWidth;
    }
  }
  return exactBitWidth;
}

} // namespace common
} // namespace concretelang