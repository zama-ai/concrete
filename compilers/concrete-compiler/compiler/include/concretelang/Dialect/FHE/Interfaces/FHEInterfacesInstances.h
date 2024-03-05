// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_CONCRETE_FHEINTERFACESINSTANCES_H
#define CONCRETELANG_DIALECT_CONCRETE_FHEINTERFACESINSTANCES_H

#include "concretelang/Dialect/FHE/Interfaces/FHEInterfaces.h"

namespace mlir {
class DialectRegistry;

namespace concretelang {
namespace FHE {
void registerFheInterfacesExternalModels(DialectRegistry &registry);
} // namespace FHE
} // namespace concretelang
} // namespace mlir

#endif
