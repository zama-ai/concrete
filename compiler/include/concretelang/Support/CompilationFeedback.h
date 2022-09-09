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
#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include "llvm/Support/Error.h"

namespace mlir {
namespace concretelang {

using StringError = ::concretelang::error::StringError;

struct CompilationFeedback {
  double complexity;

  /// @brief the total number of bytes of secret keys
  size_t totalSecretKeysSize;

  /// @brief the total number of bytes of bootstrap keys
  size_t totalBootstrapKeysSize;

  /// @brief the total number of bytes of keyswitch keys
  size_t totalKeyswitchKeysSize;

  /// @brief the total number of bytes of inputs
  size_t totalInputsSize;

  /// @brief the total number of bytes of outputs
  size_t totalOutputsSize;

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
