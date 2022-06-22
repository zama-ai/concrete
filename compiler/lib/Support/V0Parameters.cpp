// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

/// DO NOT MANUALLY EDIT THIS FILE.
/// This file was generated thanks the "parameters optimizer".
/// We should include this in our build system, but for moment it is just a cc
/// from the optimizer output.

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

#include "llvm/Support/raw_ostream.h"

#include "concrete-optimizer.hpp"
#include "concretelang/Support/V0Parameters.h"
#include <concretelang/Support/logging.h>

namespace mlir {
namespace concretelang {

optimizer::DagSolution getV0Parameter(V0FHEConstraint constraint,
                                      optimizer::Config config) {
  // the norm2 0 is equivalent to a maximum noise_factor of 2.0
  // norm2 = 0  ==>  1.0 =< noise_factor < 2.0
  // norm2 = k  ==>  2^norm2 =< noise_factor < 2.0^norm2 + 1
  double noise_factor = std::exp2(constraint.norm2 + 1);
  auto solution = concrete_optimizer::v0::optimize_bootstrap(
      constraint.p, config.security, noise_factor, config.p_error);
  return concrete_optimizer::utils::convert_to_dag_solution(solution);
}

optimizer::DagSolution getV1Parameter(optimizer::Dag &dag,
                                      optimizer::Config config) {
  return dag->optimize(config.security, config.p_error,
                       config.fallback_log_norm_woppbs);
}

static void display(V0FHEConstraint constraint,
                    optimizer::Config optimizerConfig,
                    optimizer::DagSolution sol,
                    std::chrono::milliseconds duration) {
  if (!optimizerConfig.display && !mlir::concretelang::isVerbose()) {
    return;
  }
  auto o = llvm::outs;
  o() << "--- Circuit\n"
      << "  " << constraint.p << " bits integers\n"
      << "  " << constraint.norm2 << " manp (maxi log2 norm2)\n"
      << "  " << duration.count() << "ms to solve\n"
      << "--- Optimizer config\n"
      << "  " << optimizerConfig.p_error << " error per pbs call\n"
      << "--- Complexity for each Pbs call\n"
      << "  " << (long)sol.complexity / (1000 * 1000)
      << " Millions Operations\n"
      << "--- Correctness for each Pbs call\n"
      << "  1/" << int(1.0 / sol.p_error) << " errors (" << sol.p_error << ")\n"
      << "--- Parameters resolution\n"
      << "  " << sol.glwe_dimension << "x glwe_dimension\n"
      << "  2**" << (size_t)std::log2l(sol.glwe_polynomial_size)
      << " polynomial (" << sol.glwe_polynomial_size << ")\n"
      << "  " << sol.internal_ks_output_lwe_dimension << " lwe dimension \n"
      << "  keyswitch l,b=" << sol.ks_decomposition_level_count << ","
      << sol.ks_decomposition_base_log << "\n"
      << "  blindrota l,b=" << sol.br_decomposition_level_count << ","
      << sol.br_decomposition_base_log << "\n"
      << "  wopPbs : " << (sol.use_wop_pbs ? "true" : "false") << "\n";
  if (sol.use_wop_pbs) {
    o() << "    |cb_decomp l,b=" << sol.cb_decomposition_level_count << ","
        << sol.cb_decomposition_base_log << "\n";
  }
  o() << "---\n";
}

llvm::Optional<V0Parameter> getParameter(optimizer::Description &descr,
                                         optimizer::Config config) {
  namespace chrono = std::chrono;
  auto start = chrono::high_resolution_clock::now();
  auto sol = (!descr.dag || config.strategy_v0)
                 ? getV0Parameter(descr.constraint, config)
                 : getV1Parameter(descr.dag.getValue(), config);

  auto stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
  auto duration_s = chrono::duration_cast<chrono::seconds>(duration);
  if (duration_s.count() > 3) {
    llvm::errs() << "concrete-optimizer time: " << duration_s.count() << "s\n";
  }

  display(descr.constraint, config, sol, duration);

  if (sol.p_error == 1.0) {
    // The optimizer return a p_error = 1 if there is no solution
    return llvm::None;
  }

  if (sol.use_wop_pbs) {
    llvm::errs()
        << "WARNING: a woppbs solution exists but woppbs is not available\n";
    return llvm::None;
  }

  return mlir::concretelang::V0Parameter{
      sol.glwe_dimension,
      (size_t)std::log2l(sol.glwe_polynomial_size),
      sol.internal_ks_output_lwe_dimension,
      sol.br_decomposition_level_count,
      sol.br_decomposition_base_log,
      sol.ks_decomposition_level_count,
      sol.ks_decomposition_base_log,
  };
}

} // namespace concretelang
} // namespace mlir
