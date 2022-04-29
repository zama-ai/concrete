// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
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

#include "concrete-optimizer.h"
#include "concretelang/Support/V0Parameters.h"

namespace mlir {
namespace concretelang {

static void display(V0FHEConstraint constraint,
                    optimizer::Config optimizerConfig,
                    concrete_optimizer::Solution sol,
                    std::chrono::milliseconds duration) {
  if (!optimizerConfig.display) {
    return;
  }
  auto o = llvm::outs;
  o() << "--- Circuit\n"
      << "  " << constraint.p << " bits integers\n"
      << constraint.norm2 << " manp (maxi log2 norm2)\n"

      << "--- Optimizer config\n"
      << "  " << optimizerConfig.p_error << " error per pbs call\n"

      << "--- For each Pbs call\n"
      << "  " << (long)sol.complexity / (1000 * 1000)
      << " Millions Operations\n"
      << "  1/" << int(1.0 / sol.p_error) << " errors (" << sol.p_error << ")\n"

      << "--- Parameters resolution\n"
      << "  2**" << (size_t)std::log2l(sol.glwe_polynomial_size)
      << " polynomial (" << sol.glwe_polynomial_size << ")\n"
      << "  " << sol.internal_ks_output_lwe_dimension << " lwe dimension \n"
      << "  keyswitch l,b=" << sol.ks_decomposition_level_count << ","
      << sol.ks_decomposition_base_log << "\n"
      << "  blindrota l,b=" << sol.br_decomposition_level_count << ","
      << sol.br_decomposition_base_log << "\n"
      << "  " << sol.noise_max << " variance max\n"
      << "  " << duration.count() << "ms to solve\n"
      << "---\n";
}

llvm::Optional<V0Parameter> getV0Parameter(V0FHEConstraint constraint,
                                           optimizer::Config optimizerConfig) {
  namespace chrono = std::chrono;
  int security = 128;
  // the norm2 0 is equivalent to a maximum noise_factor of 2.0
  // norm2 = 0  ==>  1.0 =< noise_factor < 2.0
  // norm2 = k  ==>  2^norm2 =< noise_factor < 2.0^norm2 + 1
  double noise_factor = std::exp2(constraint.norm2 + 1);
  // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/V0Parameters/tabulation.py#L58
  double p_error = optimizerConfig.p_error;
  auto start = chrono::high_resolution_clock::now();
  auto sol = concrete_optimizer::optimise_bootstrap(constraint.p, security,
                                                    noise_factor, p_error);
  auto stop = chrono::high_resolution_clock::now();
  if (sol.p_error == 1.0) {
    // The optimizer return a p_error = 1 if there is no solution
    return llvm::None;
  }
  auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
  auto duration_s = chrono::duration_cast<chrono::seconds>(duration);
  if (duration_s.count() > 3) {
    llvm::errs() << "concrete-optimizer time: " << duration_s.count() << "s\n";
  }

  display(constraint, optimizerConfig, sol, duration);

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
