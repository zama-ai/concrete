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
#include <optional>

#include "llvm/Support/raw_ostream.h"

#include "concrete-optimizer.hpp"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/V0Parameters.h"
#include "concretelang/Support/logging.h"

namespace mlir {
namespace concretelang {

concrete_optimizer::Options options_from_config(optimizer::Config config) {
  concrete_optimizer::Options options = {
      /* .security_level = */ config.security,
      /* .maximum_acceptable_error_probability = */ config.p_error,
      /* .default_log_norm2_woppbs = */ config.fallback_log_norm_woppbs,
      /* .use_gpu_constraints = */ config.use_gpu_constraints,
      /* .encoding = */ config.encoding,
      /* .cache_on_disk = */ config.cache_on_disk,
  };
  return options;
}

optimizer::DagSolution getV0Parameter(V0FHEConstraint constraint,
                                      optimizer::Config config) {
  // the norm2 0 is equivalent to a maximum noise_factor of 2.0
  // norm2 = 0  ==>  1.0 =< noise_factor < 2.0
  // norm2 = k  ==>  2^norm2 =< noise_factor < 2.0^norm2 + 1
  double noise_factor = std::exp2(constraint.norm2 + 1);

  auto options = options_from_config(config);

  auto solution = concrete_optimizer::v0::optimize_bootstrap(
      constraint.p, noise_factor, options);
  return concrete_optimizer::utils::convert_to_dag_solution(solution);
}

const int MAXIMUM_OPTIMIZER_CALL = 10;
optimizer::DagSolution getV1ParameterGlobalPError(optimizer::Dag &dag,
                                                  optimizer::Config config) {
  // We find the approximate translation between local and global error with a
  // calibration call
  auto ref_global_p_success = 1.0 - config.global_p_error;

  auto options = options_from_config(config);
  options.maximum_acceptable_error_probability =
      std::min(config.p_error, config.global_p_error);

  auto sol = dag->optimize(options);

  if (sol.global_p_error <= config.global_p_error) {
    // for levelled circuit the error is almost zero
    return sol;
  }
  for (int i = 2; i <= MAXIMUM_OPTIMIZER_CALL; i++) {
    auto local_p_success = 1.0 - sol.p_error;
    auto global_p_success = 1.0 - sol.global_p_error;
    auto power_global_to_local = log(local_p_success) / log(global_p_success);
    auto surrogate_p_local_success =
        pow(ref_global_p_success, power_global_to_local);

    auto surrogate_p_error = 1.0 - surrogate_p_local_success;

    // only valid when p_error is not too small and global_p_error not too high
    auto valid = 0 < surrogate_p_error && surrogate_p_error < 1.0;

    if (!valid) {
      // linear approximation, only precise for small p_error
      auto linear_correction =
          sol.p_error < 0.1 ? sol.p_error / sol.global_p_error : 0.1;
      valid = 0.0 < linear_correction && linear_correction < 1.0;
      if (!valid) {
        // global_p_error could be 0
        linear_correction = 1e-5;
      }
      surrogate_p_error =
          options.maximum_acceptable_error_probability * linear_correction;
    };

    options.maximum_acceptable_error_probability = surrogate_p_error;

    sol = dag->optimize(options);
    if (sol.global_p_error <= config.global_p_error) {
      break;
    }
  }
  return sol;
}

optimizer::DagSolution getV1Parameter(optimizer::Dag &dag,
                                      optimizer::Config config) {
  if (!std::isnan(config.global_p_error)) {
    return getV1ParameterGlobalPError(dag, config);
  }

  auto options = options_from_config(config);

  return dag->optimize(options);
}

constexpr double WARN_ABOVE_GLOBAL_ERROR_RATE = 1.0 / 1000.0;

static void display(optimizer::Description &descr,
                    optimizer::Config optimizerConfig,
                    optimizer::DagSolution sol, bool naive_user,
                    std::chrono::milliseconds duration) {
  if (!optimizerConfig.display && !mlir::concretelang::isVerbose()) {
    return;
  }
  auto constraint = descr.constraint;
  auto complexity_label =
      descr.dag ? "for the full circuit" : "for each Pbs call";
  double mops = ceil(sol.complexity / (1000 * 1000));
  auto o = llvm::outs;
  o() << "--- Circuit\n"
      << "  " << constraint.p << " bits integers\n"
      << "  " << constraint.norm2 << " manp (maxi log2 norm2)\n"
      << "  " << duration.count() << "ms to solve\n"
      << "--- User config\n"
      << "  " << optimizerConfig.p_error << " error per pbs call\n";
  if (!std::isnan(optimizerConfig.global_p_error)) {
    o() << "  " << optimizerConfig.global_p_error
        << " error per circuit call\n";
  }
  o() << "--- Complexity " << complexity_label << "\n"
      << "  " << mops << " Millions Operations\n"
      << "--- Correctness for each Pbs call\n"
      << "  1/" << int(1.0 / sol.p_error) << " errors (" << sol.p_error
      << ")\n";
  if (descr.dag && !std::isnan(sol.global_p_error)) {
    o() << "--- Correctness for the full circuit\n"
        << "  1/" << int(1.0 / sol.global_p_error) << " errors ("
        << sol.global_p_error << ")\n";
  }
  o() << "--- Parameters resolution\n"
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
        << sol.cb_decomposition_base_log << "\n"
        << "    |pp_decomp l,b=" << sol.pp_decomposition_level_count << ","
        << sol.pp_decomposition_base_log << "\n";
  }
  o() << "---\n";

  if (descr.dag && naive_user &&
      sol.global_p_error > WARN_ABOVE_GLOBAL_ERROR_RATE) {
    auto dominating_pbs =
        (int)(log(1.0 - sol.global_p_error) / log(1.0 - sol.p_error));
    o() << "---\n"
        << "!!!!! WARNING !!!!!\n"
        << "\n"
        << "HIGH ERROR RATE: 1/" << int(1.0 / sol.global_p_error)
        << " errors \n\n"
        << "Resolve by using command line option: \n"
        << "--global-error-probability=" << WARN_ABOVE_GLOBAL_ERROR_RATE
        << "\n\n"
        << "Reason:\n"
        << dominating_pbs << " pbs dominate at 1/" << int(1.0 / sol.p_error)
        << " errors rate\n";
    o() << "\n!!!!!!!!!!!!!!!!!!!\n";
  }
}

llvm::Expected<V0Parameter> getParameter(optimizer::Description &descr,
                                         CompilationFeedback &feedback,
                                         optimizer::Config config) {
  namespace chrono = std::chrono;
  auto start = chrono::high_resolution_clock::now();
  auto naive_user =
      std::isnan(config.p_error) && std::isnan(config.global_p_error);

  if (naive_user) {
    config.global_p_error = optimizer::DEFAULT_GLOBAL_P_ERROR;
  }
  if (std::isnan(config.p_error)) {
    // We always need a valid p-error
    // getV0Parameter relies only on p_error
    // getV1Parameter relies on p-error and if set global-p-error
    config.p_error = config.global_p_error;
  }

  auto sol = (!descr.dag || config.strategy_v0)
                 ? getV0Parameter(descr.constraint, config)
                 : getV1Parameter(descr.dag.value(), config);

  auto stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
  auto duration_s = chrono::duration_cast<chrono::seconds>(duration);
  if (duration_s.count() > 3) {
    llvm::errs() << "concrete-optimizer time: " << duration_s.count() << "s\n";
  }

  display(descr, config, sol, naive_user, duration);

  // The optimizer return a p_error = 1 if there is no solution
  bool no_solution = sol.p_error == 1.0;
  // The global_p_error is best effort only, so we must verify
  bool bad_solution = !std::isnan(config.global_p_error) &&
                      config.global_p_error < sol.global_p_error;

  if (no_solution || bad_solution) {
    return StreamStringError() << "Cannot find crypto parameters";
  }

  if (descr.dag && !config.display && naive_user &&
      sol.global_p_error > WARN_ABOVE_GLOBAL_ERROR_RATE) {
    llvm::errs() << "WARNING: high error rate, more details with "
                    "--display-optimizer-choice\n";
  }

  V0Parameter params;
  params.glweDimension = sol.glwe_dimension;
  params.logPolynomialSize = (size_t)std::log2l(sol.glwe_polynomial_size);
  params.nSmall = sol.internal_ks_output_lwe_dimension;
  params.brLevel = sol.br_decomposition_level_count;
  params.brLogBase = sol.br_decomposition_base_log;
  params.ksLevel = sol.ks_decomposition_level_count;
  params.ksLogBase = sol.ks_decomposition_base_log;
  params.largeInteger = std::nullopt;

  if (sol.use_wop_pbs) {
    if (sol.crt_decomposition.empty()) {
      // TODO: FIXME
      llvm::errs() << "FIXME: optimizer didn't returns the crt_decomposition\n";
      sol.crt_decomposition = {7, 8, 9, 11, 13};
    }
    LargeIntegerParameter lParams;
    for (auto m : sol.crt_decomposition) {
      lParams.crtDecomposition.push_back(m);
    }
    lParams.wopPBS.circuitBootstrap.baseLog = sol.cb_decomposition_base_log;
    lParams.wopPBS.circuitBootstrap.level = sol.cb_decomposition_level_count;
    lParams.wopPBS.packingKeySwitch.inputLweDimension =
        sol.internal_ks_output_lwe_dimension + 1;
    lParams.wopPBS.packingKeySwitch.outputPolynomialSize =
        sol.glwe_polynomial_size;
    lParams.wopPBS.packingKeySwitch.level = sol.pp_decomposition_level_count;
    lParams.wopPBS.packingKeySwitch.baseLog = sol.pp_decomposition_base_log;

    params.largeInteger = lParams;
  }

  feedback.complexity = sol.complexity;
  feedback.pError = sol.p_error;
  feedback.globalPError =
      std::isnan(sol.global_p_error) ? 0 : sol.global_p_error;

  return params;
}

} // namespace concretelang
} // namespace mlir
