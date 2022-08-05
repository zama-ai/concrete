// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_GLOBALFHECONTEXT_H_
#define CONCRETELANG_CONVERSION_GLOBALFHECONTEXT_H_
#include <cstddef>
#include <cstdint>
#include <vector>

#include "llvm/ADT/Optional.h"

namespace mlir {
namespace concretelang {

typedef std::vector<int64_t> CRTDecomposition;

struct V0FHEConstraint {
  size_t norm2;
  size_t p;
};

struct PackingKeySwitchParameter {
  size_t inputLweDimension;
  size_t outputPolynomialSize;
  size_t level;
  size_t baseLog;
};

struct CitcuitBoostrapParameter {
  size_t level;
  size_t baseLog;
};

struct WopPBSParameter {
  PackingKeySwitchParameter packingKeySwitch;
  CitcuitBoostrapParameter circuitBootstrap;
};

struct LargeIntegerParameter {
  CRTDecomposition crtDecomposition;
  WopPBSParameter wopPBS;
};

struct V0Parameter {
  size_t glweDimension;
  size_t logPolynomialSize;
  size_t nSmall;
  size_t brLevel;
  size_t brLogBase;
  size_t ksLevel;
  size_t ksLogBase;

  llvm::Optional<LargeIntegerParameter> largeInteger;

  V0Parameter() {}

  V0Parameter(size_t glweDimension, size_t logPolynomialSize, size_t nSmall,
              size_t brLevel, size_t brLogBase, size_t ksLevel,
              size_t ksLogBase)
      : glweDimension(glweDimension), logPolynomialSize(logPolynomialSize),
        nSmall(nSmall), brLevel(brLevel), brLogBase(brLogBase),
        ksLevel(ksLevel), ksLogBase(ksLogBase) {}

  // TODO remove the shift when we have true polynomial size
  size_t getPolynomialSize() { return 1 << logPolynomialSize; }

  size_t getNBigGlweDimension() { return glweDimension * getPolynomialSize(); }
};

struct V0FHEContext {
  V0FHEContext() = delete;
  V0FHEContext(const V0FHEConstraint &constraint, const V0Parameter &parameter)
      : constraint(constraint), parameter(parameter) {}

  V0FHEConstraint constraint;
  V0Parameter parameter;
};
} // namespace concretelang
} // namespace mlir

#endif
