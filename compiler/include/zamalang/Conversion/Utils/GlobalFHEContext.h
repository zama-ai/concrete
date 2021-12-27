// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_CONVERSION_GLOBALFHECONTEXT_H_
#define ZAMALANG_CONVERSION_GLOBALFHECONTEXT_H_
#include <cstddef>

namespace mlir {
namespace zamalang {

struct V0FHEConstraint {
  size_t norm2;
  size_t p;
};

struct V0Parameter {
  size_t glweDimension;
  size_t logPolynomialSize;
  size_t nSmall;
  size_t brLevel;
  size_t brLogBase;
  size_t ksLevel;
  size_t ksLogBase;

  V0Parameter() = delete;

  V0Parameter(size_t glweDimension, size_t logPolynomialSize, size_t nSmall,
              size_t brLevel, size_t brLogBase, size_t ksLevel,
              size_t ksLogBase)
      : glweDimension(glweDimension), logPolynomialSize(logPolynomialSize),
        nSmall(nSmall), brLevel(brLevel), brLogBase(brLogBase),
        ksLevel(ksLevel), ksLogBase(ksLogBase) {}

  // TODO remove the shift when we have true polynomial size
  size_t getNBigGlweDimension() {
    return glweDimension * (1 << logPolynomialSize);
  }
};

struct V0FHEContext {
  V0FHEContext() = delete;
  V0FHEContext(const V0FHEConstraint &constraint, const V0Parameter &parameter)
      : constraint(constraint), parameter(parameter) {}

  V0FHEConstraint constraint;
  V0Parameter parameter;
};
} // namespace zamalang
} // namespace mlir

#endif
