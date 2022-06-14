// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_BCONCRETE_BUFFERIZABLEOPINTERFACEIMPL_H
#define CONCRETELANG_DIALECT_BCONCRETE_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace concretelang {
namespace BConcrete {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace BConcrete
} // namespace concretelang
} // namespace mlir

#endif
