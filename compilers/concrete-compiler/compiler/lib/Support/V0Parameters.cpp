// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
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
      /* .key_sharing = */ config.key_sharing,
      /* .multi_param_strategy = */ config.multi_param_strategy,
      /* .default_log_norm2_woppbs = */ config.fallback_log_norm_woppbs,
      /* .use_gpu_constraints = */ config.use_gpu_constraints,
      /* .encoding = */ config.encoding,
      /* .cache_on_disk = */ config.cache_on_disk,
      /* .ciphertext_modulus_log = */ config.ciphertext_modulus_log,
      /* .fft_precision = */ config.fft_precision,
      /* .composable = */ config.composable,
      /* .public_keys = */ config.public_keys};
  return options;
}

optimizer::DagSolution getV0Solution(V0FHEConstraint constraint,
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

template <typename Solution, typename Optimize>
Solution getSolutionWithGlobalPError(optimizer::Config config,
                                     Optimize optimize) {
  // We find the approximate translation between local and global error with a
  // calibration call
  auto ref_global_p_success = 1.0 - config.global_p_error;

  auto options = options_from_config(config);
  options.maximum_acceptable_error_probability =
      std::min(config.p_error, config.global_p_error);

  auto sol = optimize(options);

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

    sol = optimize(options);
    if (sol.global_p_error <= config.global_p_error) {
      break;
    }
  }
  return sol;
}

optimizer::DagSolution getDagMonoSolution(optimizer::Dag &dag,
                                          optimizer::Config config) {
  auto optimize =
      [&](concrete_optimizer::Options options) -> optimizer::DagSolution {
    return dag->optimize(options);
  };
  if (!std::isnan(config.global_p_error)) {
    return getSolutionWithGlobalPError<optimizer::DagSolution>(config,
                                                               optimize);
  }
  return optimize(options_from_config(config));
}

optimizer::CircuitSolution getDagMultiSolution(optimizer::Dag &dag,
                                               optimizer::Config config) {

  auto optimize =
      [&](concrete_optimizer::Options options) -> optimizer::CircuitSolution {
    return dag->optimize_multi(options);
  };
  if (!std::isnan(config.global_p_error)) {
    return getSolutionWithGlobalPError<optimizer::CircuitSolution>(config,
                                                                   optimize);
  }
  return optimize(options_from_config(config));
}

constexpr double WARN_ABOVE_GLOBAL_ERROR_RATE = 1.0 / 1000.0;

template <typename Solution> void displaySolution(const Solution &solution);

template <> void displaySolution(const optimizer::CircuitSolution &solution) {
  llvm::errs() << "-- Circuit Solution\n";
  llvm::errs() << solution.short_dump().c_str();
}

template <> void displaySolution(const optimizer::DagSolution &sol) {
  llvm::errs() << "-- Dag Solution\n"
               << "  " << sol.glwe_dimension << "x glwe_dimension\n"
               << "  2**" << (size_t)std::log2l(sol.glwe_polynomial_size)
               << " polynomial (" << sol.glwe_polynomial_size << ")\n"
               << "  " << sol.internal_ks_output_lwe_dimension
               << " lwe dimension \n"
               << "  keyswitch l,b=" << sol.ks_decomposition_level_count << ","
               << sol.ks_decomposition_base_log << "\n"
               << "  blindrota l,b=" << sol.br_decomposition_level_count << ","
               << sol.br_decomposition_base_log << "\n"
               << "  wopPbs : " << (sol.use_wop_pbs ? "true" : "false") << "\n";
  if (sol.use_wop_pbs) {
    llvm::errs() << "    |cb_decomp l,b=" << sol.cb_decomposition_level_count
                 << "," << sol.cb_decomposition_base_log << "\n"
                 << "    |pp_decomp l,b=" << sol.pp_decomposition_level_count
                 << "," << sol.pp_decomposition_base_log << "\n";
  }
}

template <typename Solution>
void displayOptimizer(const Solution &solution,
                      const optimizer::Description &descr,
                      const optimizer::Config &config) {
  if (!config.display) {
    return;
  }
  llvm::errs() << "### Optimizer display\n";
  // Print the circuit constraint
  llvm::errs() << "--- Circuit\n"
               << "  " << descr.constraint.p << " bits integers\n"
               << "  " << descr.constraint.norm2 << " manp (maxi log2 norm2)\n";
  // Print the user configuration
  llvm::errs() << "--- User config\n"
               << "  " << config.p_error << " error per pbs call\n";
  if (!std::isnan(config.global_p_error)) {
    llvm::errs() << "  " << config.global_p_error
                 << " error per circuit call\n";
  }
  // Print the actual correctness of the solution
  llvm::errs() << "-- Solution correctness\n"
               << "  For each pbs call: "
               << " 1/" << int(1.0 / solution.p_error) << ", p_error ("
               << solution.p_error << ")\n";
  if (descr.dag && !std::isnan(solution.global_p_error)) {
    llvm::errs() << "  For the full circuit:"
                 << " 1/" << int(1.0 / solution.global_p_error)
                 << " global_p_error(" << solution.global_p_error << ")\n";
  }
  auto complexity_label =
      descr.dag ? "for the full circuit" : "for each Pbs call";
  double mops = ceil(solution.complexity / (1000 * 1000));
  llvm::errs() << "--- Complexity " << complexity_label << "\n"
               << "  " << mops << " Millions Operations\n";
  displaySolution(solution);
  llvm::errs() << "###\n";
}

//
//   if (descr.dag && naive_user &&
//       sol.global_p_error > WARN_ABOVE_GLOBAL_ERROR_RATE) {
//     auto dominating_pbs =
//         (int)(log(1.0 - sol.global_p_error) / log(1.0 - sol.p_error));
//     o() << "---\n"
//         << "!!!!! WARNING !!!!!\n"
//         << "\n"
//         << "HIGH ERROR RATE: 1/" << int(1.0 / sol.global_p_error)
//         << " errors \n\n"
//         << "Resolve by using command line option: \n"
//         << "--global-error-probability=" << WARN_ABOVE_GLOBAL_ERROR_RATE
//         << "\n\n"
//         << "Reason:\n"
//         << dominating_pbs << " pbs dominate at 1/" << int(1.0 / sol.p_error)
//         << " errors rate\n";
//     o() << "\n!!!!!!!!!!!!!!!!!!!\n";
//   }
// }

/// Convert a concrete-optimizer solution to the compiler representation
template <typename Solution> optimizer::Solution convertSolution(Solution);

/// Convert a `DagSolution` to the compiler representation
template <> optimizer::Solution convertSolution(optimizer::DagSolution sol) {
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
    assert(!sol.crt_decomposition.empty());
    LargeIntegerParameter lParams;
    for (auto m : sol.crt_decomposition) {
      lParams.crtDecomposition.push_back(m);
    }
    lParams.wopPBS.circuitBootstrap.baseLog = sol.cb_decomposition_base_log;
    lParams.wopPBS.circuitBootstrap.level = sol.cb_decomposition_level_count;
    lParams.wopPBS.packingKeySwitch.inputLweDimension =
        sol.internal_ks_output_lwe_dimension;
    lParams.wopPBS.packingKeySwitch.outputPolynomialSize =
        sol.glwe_polynomial_size;
    lParams.wopPBS.packingKeySwitch.level = sol.pp_decomposition_level_count;
    lParams.wopPBS.packingKeySwitch.baseLog = sol.pp_decomposition_base_log;

    params.largeInteger = lParams;
  }
  return optimizer::Solution(params);
}

/// Convert a `DagSolution` to the compiler representation
template <>
optimizer::Solution convertSolution(optimizer::CircuitSolution sol) {
  // Note: For now we don't support CircuitSolution with crt decomposition in
  // the pipeline
  assert(sol.crt_decomposition.empty());
  return optimizer::Solution(sol);
}

/// Fill the compilation `feedback` from a `solution` returned by the optmizer.
template <typename Solution>
void fillFeedback(Solution solution, ProgramCompilationFeedback &feedback) {
  feedback.complexity = solution.complexity;
  feedback.pError = solution.p_error;
  feedback.globalPError =
      std::isnan(solution.global_p_error) ? 0 : solution.global_p_error;
}

template <typename Solution> std::string getErrorMessage(Solution solution);

template <> std::string getErrorMessage(optimizer::CircuitSolution sol) {
  return sol.error_msg.c_str();
}

template <> std::string getErrorMessage(optimizer::DagSolution sol) {
  return "NoParametersFound";
}

/// Check if the solution p_error and global_p_error of the given `solution`
/// match the one expected by the `config`.
template <typename Solution>
llvm::Error checkPErrorSolution(Solution solution, optimizer::Config config) {
  // The optimizer return a p_error = 1 if there is no solution
  bool no_solution = solution.p_error == 1.0;
  // The global_p_error is best effort only, so we must verify
  bool bad_solution = !std::isnan(config.global_p_error) &&
                      config.global_p_error < solution.global_p_error;

  if (no_solution || bad_solution) {
    return StreamStringError() << getErrorMessage(solution);
  }

  bool naive_config = (std::isnan(config.global_p_error) &&
                       config.p_error <= WARN_ABOVE_GLOBAL_ERROR_RATE);
  if (!config.display && naive_config &&
      solution.global_p_error > WARN_ABOVE_GLOBAL_ERROR_RATE) {
    llvm::errs() << "WARNING: high error rate, more details with "
                    "--display-optimizer-choice\n";
  }

  return llvm::Error::success();
}

/// Check and convert a `solution` returned by the optimizer to the
/// optimizer::Solution, and fill the `feedback`.
template <typename Solution>
llvm::Expected<optimizer::Solution>
toCompilerSolution(Solution solution, ProgramCompilationFeedback &feedback,
                   optimizer::Config config) {
  // display(descr, config, sol, naive_user, duration);
  if (auto err = checkPErrorSolution(solution, config); err) {
    return std::move(err);
  }
  fillFeedback(solution, feedback);
  return convertSolution(solution);
}

// Returns an empty solution for non fhe programs
optimizer::Solution emptySolution() {
  optimizer::CircuitSolution solution;
  solution.is_feasible = true;
  solution.complexity = 0.;
  solution.global_p_error = 0;
  solution.p_error = 0;
  return solution;
}

llvm::Expected<optimizer::Solution>
getSolution(optimizer::Description &descr, ProgramCompilationFeedback &feedback,
            optimizer::Config config) {
  namespace chrono = std::chrono;
  // auto start = chrono::high_resolution_clock::now();
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

  // This happens for programs without fhe computation
  if (!descr.dag) {
    if (config.display) {
      llvm::errs() << "No fhe DAG, skip crypto optimization\n";
    }
    return emptySolution();
  }

  switch (config.strategy) {
  case optimizer::Strategy::V0: {
    auto sol = getV0Solution(descr.constraint, config);
    displayOptimizer(sol, descr, config);
    return toCompilerSolution(sol, feedback, config);
  }
  case optimizer::Strategy::DAG_MONO: {
    assert(descr.dag.has_value());
    auto sol = getDagMonoSolution(descr.dag.value(), config);
    displayOptimizer(sol, descr, config);
    return toCompilerSolution(sol, feedback, config);
  }
  case optimizer::Strategy::DAG_MULTI: {
    assert(descr.dag.has_value());
    auto encoding = config.encoding;
    if (encoding != concrete_optimizer::Encoding::Crt) {
      config.encoding = concrete_optimizer::Encoding::Native;
      auto sol = getDagMultiSolution(descr.dag.value(), config);
      if (sol.is_feasible || config.composable) {
        displayOptimizer(sol, descr, config);
        return toCompilerSolution(sol, feedback, config);
      }
    }
    config.strategy = optimizer::Strategy::DAG_MONO;
    config.encoding = encoding;
    return getSolution(descr, feedback, config);
  }
  }
  return StreamStringError("Unknown strategy: ") << config.strategy;
  //  auto stop = chrono::high_resolution_clock::now();
  //  auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
  //  auto duration_s = chrono::duration_cast<chrono::seconds>(duration);
  //  if (duration_s.count() > 3) {
  //    llvm::errs() << "concrete-optimizer time: " << duration_s.count() <<
  //    "s\n";
  //  }
}

} // namespace concretelang
} // namespace mlir
