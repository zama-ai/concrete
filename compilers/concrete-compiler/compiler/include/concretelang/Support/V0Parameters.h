// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_V0Parameter_H_
#define CONCRETELANG_SUPPORT_V0Parameter_H_

#include <variant>

#include "llvm/ADT/Optional.h"

#include "concrete-optimizer.hpp"
#include "concretelang/Support/CompilationFeedback.h"

namespace mlir {
namespace concretelang {

/// FHE constraint to solve using the V0 crypto optimization
/// strategy.
struct V0FHEConstraint {
  size_t norm2;
  size_t p;
};

typedef std::vector<int64_t> CRTDecomposition;

struct PackingKeySwitchParameter {
  size_t inputLweDimension;
  size_t outputPolynomialSize;
  size_t level;
  size_t baseLog;
};

struct CitcuitBoostrapParameter {
  size_t level;
  size_t baseLog;
};

struct WopPBSParameter {
  PackingKeySwitchParameter packingKeySwitch;
  CitcuitBoostrapParameter circuitBootstrap;
};

struct LargeIntegerParameter {
  CRTDecomposition crtDecomposition;
  WopPBSParameter wopPBS;
};

struct V0Parameter {
  size_t glweDimension;
  size_t logPolynomialSize;
  size_t nSmall;
  size_t brLevel;
  size_t brLogBase;
  size_t ksLevel;
  size_t ksLogBase;

  std::optional<LargeIntegerParameter> largeInteger;

  // TODO remove the shift when we have true polynomial size
  size_t getPolynomialSize() const { return 1 << logPolynomialSize; }

  size_t getNBigLweDimension() const {
    return glweDimension * getPolynomialSize();
  }
};

namespace optimizer {
constexpr double DEFAULT_GLOBAL_P_ERROR = 1.0 / 100000.0;
constexpr double UNSPECIFIED_P_ERROR = NAN; // will use DEFAULT_GLOBAL_P_ERROR
constexpr double UNSPECIFIED_GLOBAL_P_ERROR =
    NAN; // will use DEFAULT_GLOBAL_P_ERROR
constexpr uint DEFAULT_SECURITY = 128;
constexpr uint DEFAULT_FALLBACK_LOG_NORM_WOPPBS = 8;
constexpr bool DEFAULT_DISPLAY = false;
constexpr bool DEFAULT_USE_GPU_CONSTRAINTS = false;
constexpr concrete_optimizer::Encoding DEFAULT_ENCODING =
    concrete_optimizer::Encoding::Auto;
constexpr bool DEFAULT_CACHE_ON_DISK = true;
constexpr uint32_t DEFAULT_CIPHERTEXT_MODULUS_LOG = 64;
constexpr uint32_t DEFAULT_FFT_PRECISION = 53;

/// The strategy of the crypto optimization
enum Strategy {
  /// V0 is a strategy based on the worst case atomic pattern
  V0 = 0,
  /// DAG_MONO is a strategy that used the optimizer dag but resolve with a
  /// unique set of keyswitch and boostrap key
  DAG_MONO,
  /// DAG_MULTI is a strategy that used the optimizer dag but resolve with a
  /// multiple set of keyswitch and boostrap key
  DAG_MULTI
};

std::string const StrategyLabel[] = {"V0", "dag-mono", "dag-multi"};

constexpr Strategy DEFAULT_STRATEGY = Strategy::DAG_MONO;
constexpr bool DEFAULT_KEY_SHARING = true;

struct Config {
  double p_error;
  double global_p_error;
  bool display;
  Strategy strategy;
  bool key_sharing;
  std::uint64_t security;
  double fallback_log_norm_woppbs;
  bool use_gpu_constraints;
  concrete_optimizer::Encoding encoding;
  bool cache_on_disk;
  uint32_t ciphertext_modulus_log;
  uint32_t fft_precision;
};

constexpr Config DEFAULT_CONFIG = {
    UNSPECIFIED_P_ERROR,
    UNSPECIFIED_GLOBAL_P_ERROR,
    DEFAULT_DISPLAY,
    DEFAULT_STRATEGY,
    DEFAULT_KEY_SHARING,
    DEFAULT_SECURITY,
    DEFAULT_FALLBACK_LOG_NORM_WOPPBS,
    DEFAULT_USE_GPU_CONSTRAINTS,
    DEFAULT_ENCODING,
    DEFAULT_CACHE_ON_DISK,
    DEFAULT_CIPHERTEXT_MODULUS_LOG,
    DEFAULT_FFT_PRECISION,
};

using Dag = rust::Box<concrete_optimizer::OperationDag>;
using DagSolution = concrete_optimizer::dag::DagSolution;
using CircuitSolution = concrete_optimizer::dag::CircuitSolution;

/* Contains any circuit description usable by the concrete-optimizer */
struct Description {
  V0FHEConstraint constraint;
  std::optional<optimizer::Dag> dag;
};

/// The Solution is a variant of a V0Parameter or a CircuitSolution depending of
/// optimizer config.
typedef std::variant<V0Parameter, CircuitSolution> Solution;

} // namespace optimizer

struct CompilationFeedback;

llvm::Expected<optimizer::Solution>
getSolution(optimizer::Description &descr, CompilationFeedback &feedback,
            optimizer::Config optimizerConfig);

// As for now the solution which contains a crt encoding is mono parameter only
// we have some parts of the pipeline that rely on that.
// TODO: Remove this function
inline std::optional<CRTDecomposition>
getCrtDecompositionFromSolution(optimizer::Solution solution) {
  if (auto mono = std::get_if<V0Parameter>(&solution); mono != nullptr) {
    if (mono->largeInteger.has_value()) {
      return mono->largeInteger->crtDecomposition;
    }
  }
  // TODO: Integrate the CircuitSolution with crt
  return std::nullopt;
}

// Temporary function for hack FHEToScalar
// TODO: Remove this function
inline size_t getPolynomialSizeFromSolution(optimizer::Solution solution) {
  if (auto mono = std::get_if<V0Parameter>(&solution); mono != nullptr) {
    return mono->getPolynomialSize();
  }
  return 42;
}

} // namespace concretelang
} // namespace mlir

static inline std::string toString(mlir::concretelang::optimizer::Strategy s) {
  if (s <= mlir::concretelang::optimizer::DAG_MULTI)
    return mlir::concretelang::optimizer::StrategyLabel[s];
  else
    return "unknown";
}

static inline std::ostream &
operator<<(std::ostream &OS, mlir::concretelang::optimizer::Strategy s) {
  return OS << toString(s);
}

#endif
