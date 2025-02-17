#include "concretelang/Dialect/GLWE/IR/GLWEOps.h"
#include "concretelang/Dialect/GLWE/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace concretelang {
namespace GLWE {

NoiseDistribution DEFAULT_NOISE_DISTRIBUTION = NoiseDistribution::Gaussian;
int DEFAULT_SECURITY_LEVEL = 128;
namespace {
class InjectDefaultVariancesPass
    : public GLWEInjectDefaultVariancesBase<InjectDefaultVariancesPass> {
public:
  void runOnOperation() override {
    auto op = getOperation();

    // Get default module security level and noise distribution
    auto defaultSecurityLevel = DEFAULT_SECURITY_LEVEL;
    if (auto securityAttr =
            op->getAttr("glwe.security").dyn_cast_or_null<IntegerAttr>()) {
      defaultSecurityLevel = securityAttr.getValue().getSExtValue();
    }
    auto defaultNoiseDistribution = DEFAULT_NOISE_DISTRIBUTION;
    if (auto noiseDistributionAttr =
            op->getAttr("glwe.noise_distribution")
                .dyn_cast_or_null<NoiseDistributionAttr>()) {
      defaultNoiseDistribution = noiseDistributionAttr.getValue();
    }

    op->walk([&](func::FuncOp func) {
      mlir::SmallVector<mlir::Type> newInputs;
      for (auto x : llvm::enumerate(func.getArguments())) {
        if (auto inter = x.value().getType().dyn_cast<GLWETypeInterface>()) {
          // Get security level and noise distribution for the argument
          auto securityLevel = defaultSecurityLevel;
          if (auto securityAttr = func.getArgAttr(x.index(), "glwe.security")
                                      .dyn_cast_or_null<IntegerAttr>()) {
            securityLevel = securityAttr.getValue().getSExtValue();
          }
          // Get noise distribution attribute or default one
          auto noiseDistribution = defaultNoiseDistribution;
          if (auto noiseDistributionAttr =
                  func.getArgAttr(x.index(), "glwe.noise_distribution")
                      .dyn_cast_or_null<NoiseDistributionAttr>()) {
            noiseDistribution = noiseDistributionAttr.getValue();
          }
          // Get the minimal variance thanks type interface
          auto withMinVariance =
              inter.withMinimalVariance(noiseDistribution, securityLevel);
          x.value().setType(withMinVariance);
        }
        newInputs.push_back(x.value().getType());
      }
      auto newFuncType = mlir::FunctionType::get(&getContext(), newInputs,
                                                 func.getResultTypes());
      func.setType(newFuncType);
    });
    op->walk([&](GLWEOpInterface glweOp) {
      if (!glweOp->getAttr("variance")) {
        glweOp->setAttr(
            "variance",
            GLWEExprAttr::get(&getContext(), glweOp.defaultVariance()));
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createInjectDefaultVariances() {
  return std::make_unique<InjectDefaultVariancesPass>();
}
} // namespace GLWE
} // namespace concretelang
} // namespace mlir