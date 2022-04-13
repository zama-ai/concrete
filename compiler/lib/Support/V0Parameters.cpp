// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

/// DO NOT MANUALLY EDIT THIS FILE.
/// This file was generated thanks the "parameters optimizer".
/// We should include this in our build system, but for moment it is just a cc
/// from the optimizer output.

#include <cassert>
#include <cmath>
#include <iostream>

#include "concrete-optimizer.h"
#include "concretelang/Support/V0Parameters.h"

namespace mlir {
namespace concretelang {

const double P_ERROR_4_SIGMA = 1.0 - 0.999936657516;

llvm::Optional<V0Parameter> getV0Parameter(V0FHEConstraint constraint) {
  int security = 128;
  // the norm2 0 is equivalent to a maximum noise_factor of 2.0
  // norm2 = 0  ==>  1.0 =< noise_factor < 2.0
  // norm2 = k  ==>  2^norm2 =< noise_factor < 2.0^norm2 + 1
  double noise_factor = std::exp2(constraint.norm2 + 1);
  // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/V0Parameters/tabulation.py#L58
  double p_error = P_ERROR_4_SIGMA;
  auto sol = concrete_optimizer::optimise_bootstrap(constraint.p, security,
                                                    noise_factor, p_error);

  if (sol.p_error == 1.0) {
    // The optimizer return a p_error = 1 if there is no solution
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
