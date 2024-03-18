// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
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

char batched_add_eint[] = "batched_add_eint";
char batched_add_eint_int[] = "batched_add_eint_int";
char batched_add_eint_int_cst[] = "batched_add_eint_int_cst";
char batched_mul_eint_int[] = "batched_mul_eint_int";
char batched_mul_eint_int_cst[] = "batched_mul_eint_int_cst";
char batched_neg_eint[] = "batched_neg_eint";
char batched_keyswitch[] = "batched_keyswitch";
char batched_bootstrap[] = "batched_bootstrap";
char batched_mapped_bootstrap[] = "batched_mapped_bootstrap";
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
      auto outType =
          op->getResult(0).getType().dyn_cast_or_null<mlir::TensorType>();
      auto outSize = outType.getDimSize(outType.getRank() - 1);
      auto attrList = mlir::NamedAttrList(op->getAttrs());
      attrList.append("output_size", builder.getI32IntegerAttr(outSize));
      llvm::SmallVector<mlir::NamedAttribute> combinedAttrs =
          llvm::to_vector(attrList);

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

    mlir::concretelang::Concrete::BatchedAddLweTensorOp::attachInterface<
        ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::BatchedAddLweTensorOp,
            batched_add_eint>>(*ctx);
    mlir::concretelang::Concrete::BatchedAddPlaintextLweTensorOp::
        attachInterface<ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::BatchedAddPlaintextLweTensorOp,
            batched_add_eint_int>>(*ctx);
    mlir::concretelang::Concrete::BatchedAddPlaintextCstLweTensorOp::
        attachInterface<ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::BatchedAddPlaintextCstLweTensorOp,
            batched_add_eint_int_cst>>(*ctx);
    mlir::concretelang::Concrete::BatchedMulCleartextLweTensorOp::
        attachInterface<ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::BatchedMulCleartextLweTensorOp,
            batched_mul_eint_int>>(*ctx);
    mlir::concretelang::Concrete::BatchedMulCleartextCstLweTensorOp::
        attachInterface<ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::BatchedMulCleartextCstLweTensorOp,
            batched_mul_eint_int_cst>>(*ctx);
    mlir::concretelang::Concrete::BatchedNegateLweTensorOp::attachInterface<
        ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::BatchedNegateLweTensorOp,
            batched_neg_eint>>(*ctx);
    mlir::concretelang::Concrete::BatchedKeySwitchLweTensorOp::attachInterface<
        ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::BatchedKeySwitchLweTensorOp,
            batched_keyswitch, true>>(*ctx);
    mlir::concretelang::Concrete::BatchedBootstrapLweTensorOp::attachInterface<
        ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::BatchedBootstrapLweTensorOp,
            batched_bootstrap, true>>(*ctx);
    mlir::concretelang::Concrete::BatchedMappedBootstrapLweTensorOp::
        attachInterface<ReplaceWithProcessSDFGConversionInterface<
            mlir::concretelang::Concrete::BatchedMappedBootstrapLweTensorOp,
            batched_mapped_bootstrap, true>>(*ctx);
  });
}
} // namespace SDFG
} // namespace concretelang
} // namespace mlir
