// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_BITS_SIZE_H
#define CONCRETELANG_COMMON_BITS_SIZE_H

#include <stdlib.h>

namespace concretelang {
namespace common {

size_t bitWidthAsWord(size_t exactBitWidth);

}
} // namespace concretelang

#endif