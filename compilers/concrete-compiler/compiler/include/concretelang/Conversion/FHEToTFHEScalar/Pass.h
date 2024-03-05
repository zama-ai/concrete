// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_FHETOTFHESCALAR_PASS_H_
#define CONCRETELANG_CONVERSION_FHETOTFHESCALAR_PASS_H_

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Casting.h"
#include <list>

namespace mlir {
namespace concretelang {

struct ScalarLoweringParameters {
  size_t polynomialSize;
  ScalarLoweringParameters(size_t polySize) : polynomialSize(polySize){};
};

/// Create a pass to convert `FHE` dialect to `TFHE` dialect with the scalar
// strategy.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertFHEToTFHEScalarPass(ScalarLoweringParameters loweringParameters);
} // namespace concretelang
} // namespace mlir

#endif
