// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.


#ifndef CONCRETELANG_CONVERSION_HLFHETENSOROPSTOLINALG_PASS_H_
#define CONCRETELANG_CONVERSION_HLFHETENSOROPSTOLINALG_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace concretelang {
/// Create a pass to convert `HLFHE` tensor operators to linal.generic
/// operators.
std::unique_ptr<mlir::FunctionPass> createConvertHLFHETensorOpsToLinalg();
} // namespace concretelang
} // namespace mlir

#endif