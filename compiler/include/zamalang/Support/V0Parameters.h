// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_SUPPORT_V0Parameter_H_
#define ZAMALANG_SUPPORT_V0Parameter_H_

#include "zamalang/Conversion/Utils/GlobalFHEContext.h"
#include <cstddef>

namespace mlir {
namespace zamalang {

const V0Parameter *getV0Parameter(V0FHEConstraint constraint);

} // namespace zamalang
} // namespace mlir
#endif
