#ifndef ZAMALANG_SUPPORT_V0Parameter_H_
#define ZAMALANG_SUPPORT_V0Parameter_H_

#include "zamalang/Conversion/Utils/GlobalFHEContext.h"
#include <cstddef>

namespace mlir {
namespace zamalang {

V0Parameter *getV0Parameter(V0FHEConstraint constraint);

} // namespace zamalang
} // namespace mlir
#endif