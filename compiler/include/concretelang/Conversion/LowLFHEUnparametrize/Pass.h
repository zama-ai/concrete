// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.


#ifndef CONCRETELANG_CONVERSION_LOWLFHEUNPARAMETRIZE_PASS_H_
#define CONCRETELANG_CONVERSION_LOWLFHEUNPARAMETRIZE_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLowLFHEUnparametrizePass();
} // namespace concretelang
} // namespace mlir

#endif