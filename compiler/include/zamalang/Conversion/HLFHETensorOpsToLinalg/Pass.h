// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.


#ifndef ZAMALANG_CONVERSION_HLFHETENSOROPSTOLINALG_PASS_H_
#define ZAMALANG_CONVERSION_HLFHETENSOROPSTOLINALG_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace zamalang {
/// Create a pass to convert `HLFHE` tensor operators to linal.generic
/// operators.
std::unique_ptr<mlir::FunctionPass> createConvertHLFHETensorOpsToLinalg();
} // namespace zamalang
} // namespace mlir

#endif