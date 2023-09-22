// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_COMPILATIONFEEDBACK_H_
#define CONCRETELANG_SUPPORT_COMPILATIONFEEDBACK_H_

#include <cstddef>
#include <vector>

#include "boost/outcome.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Protocol.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace protocol = concreteprotocol;
using concretelang::protocol::Message;

namespace mlir {
namespace concretelang {

using StringError = ::concretelang::error::StringError;

enum class PrimitiveOperation {
  PBS,
  WOP_PBS,
  KEY_SWITCH,
  CLEAR_ADDITION,
  ENCRYPTED_ADDITION,
  CLEAR_MULTIPLICATION,
  ENCRYPTED_NEGATION,
};

enum class KeyType {
  SECRET,
  BOOTSTRAP,
  KEY_SWITCH,
  PACKING_KEY_SWITCH,
};

struct Statistic {
  std::string location;
  PrimitiveOperation operation;
  std::vector<std::pair<KeyType, size_t>> keys;
  std::optional<int64_t> count;
};

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

  /// @brief statistics
  std::vector<Statistic> statistics;

  /// @brief memory usage per location
  std::map<std::string, std::optional<int64_t>> memoryUsagePerLoc;

  /// Fill the sizes from the program info.
  void fillFromProgramInfo(const Message<protocol::ProgramInfo> &params);

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
