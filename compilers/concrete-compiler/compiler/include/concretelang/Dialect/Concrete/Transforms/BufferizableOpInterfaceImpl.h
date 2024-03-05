// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_CONCRETE_BUFFERIZABLEOPINTERFACEIMPL_H
#define CONCRETELANG_DIALECT_CONCRETE_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace concretelang {
namespace Concrete {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace Concrete
} // namespace concretelang
} // namespace mlir

#endif
