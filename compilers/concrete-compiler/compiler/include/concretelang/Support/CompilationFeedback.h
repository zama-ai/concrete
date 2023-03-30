// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_COMPILATIONFEEDBACK_H_
#define CONCRETELANG_SUPPORT_COMPILATIONFEEDBACK_H_

#include <cstddef>
#include <vector>

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/Common/Error.h"
#include "llvm/Support/Error.h"

namespace mlir {
namespace concretelang {

using StringError = ::concretelang::error::StringError;

struct CompilationFeedback {
  double complexity;

  /// @brief Probability of error for every PBS.
  double pError;

  /// @brief Probability of error for the whole programs.
  double globalPError;

  /// @brief the total number of bytes of secret keys
  uint64_t totalSecretKeysSize;

  /// @brief the total number of bytes of bootstrap keys
  uint64_t totalBootstrapKeysSize;

  /// @brief the total number of bytes of keyswitch keys
  uint64_t totalKeyswitchKeysSize;

  /// @brief the total number of bytes of inputs
  uint64_t totalInputsSize;

  /// @brief the total number of bytes of outputs
  uint64_t totalOutputsSize;

  /// @brief crt decomposition of outputs, if crt is not used, empty vectors
  std::vector<std::vector<int64_t>> crtDecompositionsOfOutputs;

  /// Fill the sizes from the client parameters.
  void
  fillFromClientParameters(::concretelang::clientlib::ClientParameters params);

  /// Load the compilation feedback from a path
  static outcome::checked<CompilationFeedback, StringError>
  load(std::string path);
};

llvm::json::Value toJSON(const mlir::concretelang::CompilationFeedback &);
bool fromJSON(const llvm::json::Value,
              mlir::concretelang::CompilationFeedback &, llvm::json::Path);

} // namespace concretelang
} // namespace mlir

static inline llvm::raw_ostream &
operator<<(llvm::raw_string_ostream &OS,
           mlir::concretelang::CompilationFeedback cp) {
  return OS << llvm::formatv("{0:2}", toJSON(cp));
}
#endif
