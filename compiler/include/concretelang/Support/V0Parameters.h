// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_V0Parameter_H_
#define CONCRETELANG_SUPPORT_V0Parameter_H_

#include "llvm/ADT/Optional.h"

#include "concrete-optimizer.hpp"
#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include "concretelang/Support/CompilationFeedback.h"

namespace mlir {
namespace concretelang {

namespace optimizer {
constexpr double P_ERROR_4_SIGMA = 1.0 - 0.999936657516;
constexpr double UNSPECIFIED_P_ERROR = NAN; // will use the default p error
constexpr double NO_GLOBAL_P_ERROR = NAN;   // will fallback on p error
constexpr uint DEFAULT_SECURITY = 128;
constexpr uint DEFAULT_FALLBACK_LOG_NORM_WOPPBS = 8;
constexpr bool DEFAULT_DISPLAY = false;
constexpr bool DEFAULT_STARTEGY_V0 = false;

struct Config {
  double p_error;
  double global_p_error;
  bool display;
  bool strategy_v0;
  std::uint64_t security;
  double fallback_log_norm_woppbs;
};

constexpr Config DEFAULT_CONFIG = {
    UNSPECIFIED_P_ERROR, NO_GLOBAL_P_ERROR, DEFAULT_DISPLAY,
    DEFAULT_STARTEGY_V0, DEFAULT_SECURITY,  DEFAULT_FALLBACK_LOG_NORM_WOPPBS};

using Dag = rust::Box<concrete_optimizer::OperationDag>;
using Solution = concrete_optimizer::v0::Solution;
using DagSolution = concrete_optimizer::dag::DagSolution;

/* Contains any circuit description usable by the concrete-optimizer */
struct Description {
  V0FHEConstraint constraint;
  llvm::Optional<optimizer::Dag> dag;
};

} // namespace optimizer

llvm::Expected<V0Parameter> getParameter(optimizer::Description &descr,
                                         CompilationFeedback &feedback,
                                         optimizer::Config optimizerConfig);
} // namespace concretelang
} // namespace mlir
#endif
