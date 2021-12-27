// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.


#ifndef ZAMALANG_CONVERSION_HLFHETOMIDLFHE_PASS_H_
#define ZAMALANG_CONVERSION_HLFHETOMIDLFHE_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace zamalang {
/// Create a pass to convert `HLFHE` dialect to `MidLFHE` dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertHLFHEToMidLFHEPass();
} // namespace zamalang
} // namespace mlir

#endif