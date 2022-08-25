// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "concretelang/Conversion/Passes.h"

namespace {
struct BConcreteToCAPIPass : public BConcreteToCAPIBase<BConcreteToCAPIPass> {
  void runOnOperation() final;
};
} // namespace

void BConcreteToCAPIPass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  mlir::RewritePatternSet patterns(&getContext());

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertBConcreteToCAPIPass() {
  return std::make_unique<BConcreteToCAPIPass>();
}
} // namespace concretelang
} // namespace mlir
