// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.


#ifndef ZAMALANG_CONVERSION_MIDLFHEGLOBALPARAMETRIZATION_PASS_H_
#define ZAMALANG_CONVERSION_MIDLFHEGLOBALPARAMETRIZATION_PASS_H_

#include "mlir/Pass/Pass.h"

#include "zamalang/Conversion/Utils/GlobalFHEContext.h"

namespace mlir {
namespace zamalang {
/// Create a pass to inject fhe parameters to the MidLFHE types and operators.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertMidLFHEGlobalParametrizationPass(
    mlir::zamalang::V0FHEContext &fheContext);
} // namespace zamalang
} // namespace mlir

#endif