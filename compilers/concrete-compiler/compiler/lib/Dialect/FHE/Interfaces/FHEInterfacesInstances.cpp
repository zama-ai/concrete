// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/FHE/Interfaces/FHEInterfacesInstances.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/Interfaces/FHEInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace concretelang {
namespace FHE {

using namespace mlir::tensor;

void registerFheInterfacesExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TensorDialect *dialect) {
    ExtractOp::attachInterface<UnaryEint>(*ctx);
    InsertSliceOp::attachInterface<MaxNoise>(*ctx);
    InsertOp::attachInterface<MaxNoise>(*ctx);
    ParallelInsertSliceOp::attachInterface<MaxNoise>(*ctx);
  });
}
} // namespace FHE
} // namespace concretelang
} // namespace mlir
