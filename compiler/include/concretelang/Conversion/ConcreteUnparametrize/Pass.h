// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_CONCRETEUNPARAMETRIZE_PASS_H_
#define CONCRETELANG_CONVERSION_CONCRETEUNPARAMETRIZE_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertConcreteUnparametrizePass();
} // namespace concretelang
} // namespace mlir

#endif