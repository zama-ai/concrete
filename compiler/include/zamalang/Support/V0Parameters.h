#ifndef ZAMALANG_SUPPORT_V0Parameter_H_
#define ZAMALANG_SUPPORT_V0Parameter_H_

#include <cstddef>

namespace mlir {
namespace zamalang {

typedef struct V0Parameter {
  size_t k;
  size_t polynomialSize;
  size_t nSmall;
  size_t brLevel;
  size_t brLogBase;
  size_t ksLevel;
  size_t ksLogBase;

  V0Parameter() {}

  V0Parameter(size_t k, size_t polynomialSize, size_t nSmall, size_t brLevel,
              size_t brLogBase, size_t ksLevel, size_t ksLogBase)
      : k(k), polynomialSize(polynomialSize), nSmall(nSmall), brLevel(brLevel),
        brLogBase(brLogBase), ksLevel(ksLevel), ksLogBase(ksLogBase) {}

} V0Parameter;

V0Parameter *getV0Parameter(size_t norm, size_t p);

} // namespace zamalang
} // namespace mlir
#endif