// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_TFHEGLOBALPARAMETRIZATION_PASS_H_
#define CONCRETELANG_CONVERSION_TFHEGLOBALPARAMETRIZATION_PASS_H_

#include "mlir/Pass/Pass.h"

#include "concretelang/Conversion/Utils/GlobalFHEContext.h"

namespace mlir {
namespace concretelang {
/// Create a pass to inject fhe parameters to the TFHE types and operators.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTFHEGlobalParametrizationPass(const V0Parameter parameter);
} // namespace concretelang
} // namespace mlir

#endif
