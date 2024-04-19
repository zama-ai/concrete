// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_FHE_ANALYSIS_CONCRETE_OPTIMIZER_H
#define CONCRETELANG_DIALECT_FHE_ANALYSIS_CONCRETE_OPTIMIZER_H

#include <map>
#include <mlir/Pass/Pass.h>

#include "concrete-optimizer.hpp"

#include "concretelang/Support/V0Parameters.h"

namespace mlir {
namespace concretelang {

namespace optimizer {
std::unique_ptr<mlir::Pass> createDagPass(optimizer::Config config,
                                          concrete_optimizer::Dag &dag);

void applyCompositionRules(optimizer::Config config,
                           concrete_optimizer::Dag &dag);

} // namespace optimizer
} // namespace concretelang
} // namespace mlir

#endif
