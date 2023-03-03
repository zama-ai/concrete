// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/SDFG/IR/SDFGDialect.h"
#include "concretelang/Dialect/SDFG/IR/SDFGOps.h"
#include "concretelang/Dialect/SDFG/Interfaces/SDFGConvertibleInterface.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace concretelang {
namespace SDFG {
namespace {
char add_eint[] = "add_eint";
char add_eint_int[] = "add_eint_int";
char mul_eint_int[] = "mul_eint_int";
char neg_eint[] = "neg_eint";
char keyswitch[] = "keyswitch";
char bootstrap[] = "bootstrap";
} // namespace

template <typename Op, char const *processName, bool copyAttributes = false>
struct ReplaceWithProcessSDFGConversionInterface
    : public SDFGConvertibleOpInterface::ExternalModel<
          ReplaceWithProcessSDFGConversionInterface<Op, processName,
                                                    copyAttributes>,
          Op> {
  MakeProcess convert(Operation *op, mlir::ImplicitLocOpBuilder &builder,
                      ::mlir::Value dfg, ::mlir::ValueRange inStreams,
                      ::mlir::ValueRange outStreams) const {
    llvm::SmallVector<mlir::Value> streams = llvm::to_vector(inStreams);
    streams.append(outStreams.begin(), outStreams.end());
    MakeProcess process = builder.create<MakeProcess>(
        *symbolizeProcessKind(processName), dfg, streams);

    if (copyAttributes) {
      llvm::SmallVector<mlir::NamedAttribute> combinedAttrs =
          llvm::to_vector(op->getAttrs());

      for (mlir::NamedAttribute attr : process->getAttrs()) {
        combinedAttrs.push_back(attr);
      }

      process->setAttrs(combinedAttrs);
    }

    return process;
  }
};

void registerSDFGConvertibleOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            Concrete::ConcreteDialect *dialect) {
    mlir::concretelang::Concrete::AddLweTensorOp::attachInterface<
        ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::AddLweTensorOp, add_eint>>(*ctx);

    mlir::concretelang::Concrete::AddPlaintextLweTensorOp::attachInterface<
        ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::AddPlaintextLweTensorOp,
            add_eint_int>>(*ctx);

    mlir::concretelang::Concrete::MulCleartextLweTensorOp::attachInterface<
        ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::MulCleartextLweTensorOp,
            mul_eint_int>>(*ctx);

    mlir::concretelang::Concrete::NegateLweTensorOp::attachInterface<
        ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::NegateLweTensorOp, neg_eint>>(*ctx);

    mlir::concretelang::Concrete::KeySwitchLweTensorOp::attachInterface<
        ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::KeySwitchLweTensorOp, keyswitch,
            true>>(*ctx);

    mlir::concretelang::Concrete::BootstrapLweTensorOp::attachInterface<
        ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::BootstrapLweTensorOp, bootstrap,
            true>>(*ctx);
  });
}
} // namespace SDFG
} // namespace concretelang
} // namespace mlir
