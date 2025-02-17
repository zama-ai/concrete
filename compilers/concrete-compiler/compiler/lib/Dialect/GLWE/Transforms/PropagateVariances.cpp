#include "concretelang/Dialect/GLWE/IR/GLWEOps.h"
#include "concretelang/Dialect/GLWE/Transforms/Transforms.h"
#include "concretelang/Support/Variants.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace concretelang {
namespace GLWE {

namespace {
class PropagateVariancesPass
    : public GLWEPropagateVariancesBase<PropagateVariancesPass> {
public:
  void runOnOperation() override {
    auto op = getOperation();
    auto getValueFromDomainAttrs = [&](llvm::StringRef symbol) {
      auto it =
          llvm::find_if(op->getAttrDictionary(), [&](NamedAttribute attr) {
            auto domainAttr = attr.getValue().dyn_cast<DomainAttr>();
            if (!domainAttr)
              return false;
            return domainAttr.getVar().getName().getValue() == symbol;
          });
      if (it == op->getAttrDictionary().end())
        return std::optional<double>();
      auto values = it->getValue().dyn_cast<DomainAttr>().getValues();
      return std::visit(
          [](auto &&values) {
            using T = std::decay_t<decltype(values)>;
            if constexpr (std::is_same_v<T, llvm::SmallVector<int64_t>>) {
              if (values.size() == 1) {
                return std::optional<double>(values[0]);
              }
              return std::optional<double>();
            }
            if constexpr (std::is_same_v<T, llvm::SmallVector<double>>) {
              if (values.size() == 1) {
                return std::optional<double>(values[0]);
              }
              return std::optional<double>();
            }
            assert(false);
          },
          values);
    };

    auto replaceSymbolWithDomain = [&](GLWEExpr e) {
      if (auto symbolExpr = e.dyn_cast<GlweSymbolExpr>()) {
        auto value = getValueFromDomainAttrs(symbolExpr.getSymbolName());
        if (value.has_value())
          return getGlweConstantExpr(value.value(), e.getContext());
        return e;
      }
      return e;
    };

    // For all function replace and simplify arguments
    op->walk([&](func::FuncOp func) {
      mlir::SmallVector<mlir::Type> newInputs;
      for (auto x : llvm::enumerate(func.getArguments())) {
        if (auto inter = x.value().getType().dyn_cast<GLWETypeInterface>()) {
          auto variance =
              inter.getVariance().replace(replaceSymbolWithDomain).simplify();
          auto withVariance = inter.withVariance(variance);
          x.value().setType(withVariance);
        }
        newInputs.push_back(x.value().getType());
      }
      func.setType(mlir::FunctionType::get(&getContext(), newInputs,
                                           func.getResultTypes()));
    });

    // For all GLWEOp propagate variance to the return type
    if (op->walk([&](GLWEOpInterface glweOp) {
            // TODO: Make assertion more robust
            assert(glweOp->getNumResults() == 1);
            auto result = glweOp->getResult(0);
            auto resultType = result.getType().dyn_cast<GLWETypeInterface>();

            assert(resultType != nullptr);
            auto outputVariance = glweOp.resolveOutputVariance();
            if (outputVariance.has_error()) {
              glweOp->emitOpError("Cannot propagate variance: " +
                                  outputVariance.error().mesg);
              return WalkResult::interrupt();
            }
            resultType =
                resultType.withVariance(outputVariance.value()
                                            .replace(replaceSymbolWithDomain)
                                            .simplify());
            result.setType(resultType);
            return WalkResult::advance();
          }).wasInterrupted()) {
      this->signalPassFailure();
      return;
    }
    // Fixing up function signature
    // TODO: Should we make that more generic?
    if (op->walk([&](func::FuncOp func) {
            llvm::SmallVector<mlir::Type> resTypes;
            for (auto &block : func.getBody().getBlocks()) {
              if (!resTypes.empty()) {
                if (resTypes != llvm::SmallVector<mlir::Type>(
                                    block.getTerminator()->getOperandTypes())) {
                  func.emitOpError("Several return op with different types");
                  return WalkResult::interrupt();
                }
              }
              resTypes.append(block.getTerminator()->getOperandTypes().begin(),
                              block.getTerminator()->getOperandTypes().end());
            }
            func.setType(mlir::FunctionType::get(
                &getContext(), func.getArgumentTypes(), resTypes));
            return WalkResult::advance();
          }).wasInterrupted()) {
      this->signalPassFailure();
      return;
    }
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createPropagateVariances() {
  return std::make_unique<PropagateVariancesPass>();
}
} // namespace GLWE
} // namespace concretelang
} // namespace mlir