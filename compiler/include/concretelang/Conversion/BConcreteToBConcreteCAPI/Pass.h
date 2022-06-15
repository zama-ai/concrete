// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_BCONCRETETOBCONCRETECAPI_PASS_H_
#define CONCRETELANG_CONVERSION_BCONCRETETOBCONCRETECAPI_PASS_H_

#include "mlir/Pass/Pass.h"

#include "concretelang/Conversion/Utils/GlobalFHEContext.h"

namespace mlir {
namespace concretelang {
/// Create a pass to convert `Concrete` operators to function call to the
/// `ConcreteCAPI`
std::unique_ptr<OperationPass<ModuleOp>>
createConvertBConcreteToBConcreteCAPIPass();
} // namespace concretelang
} // namespace mlir

#endif
