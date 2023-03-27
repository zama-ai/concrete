// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_ENCODINGS_H_
#define CONCRETELANG_SUPPORT_ENCODINGS_H_

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "boost/outcome.h"
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"

namespace mlir {
namespace concretelang {
namespace encodings {

/// Represents the encoding of a small (unchunked) `FHE::eint` type.
struct EncryptedIntegerScalarEncoding {
  uint64_t width;
  bool isSigned;
};
bool fromJSON(const llvm::json::Value, EncryptedIntegerScalarEncoding &,
              llvm::json::Path);
llvm::json::Value toJSON(const EncryptedIntegerScalarEncoding &);
static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                            EncryptedIntegerScalarEncoding e) {
  return OS << llvm::formatv("{0:2}", toJSON(e));
}

/// Represents the encoding of a big (chunked) `FHE::eint` type.
struct EncryptedChunkedIntegerScalarEncoding {
  uint64_t width;
  bool isSigned;
  uint64_t chunkSize;
  uint64_t chunkWidth;
};
bool fromJSON(const llvm::json::Value, EncryptedChunkedIntegerScalarEncoding &,
              llvm::json::Path);
llvm::json::Value toJSON(const EncryptedChunkedIntegerScalarEncoding &);
static inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &OS, EncryptedChunkedIntegerScalarEncoding e) {
  return OS << llvm::formatv("{0:2}", toJSON(e));
}

/// Represents the encoding of a `FHE::ebool` type.
struct EncryptedBoolScalarEncoding {};
bool fromJSON(const llvm::json::Value, EncryptedBoolScalarEncoding &,
              llvm::json::Path);
llvm::json::Value toJSON(const EncryptedBoolScalarEncoding &);
static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                            EncryptedBoolScalarEncoding e) {
  return OS << llvm::formatv("{0:2}", toJSON(e));
}

/// Represents the encoding of a builtin integer type.
struct PlaintextScalarEncoding {
  uint64_t width;
};
bool fromJSON(const llvm::json::Value, PlaintextScalarEncoding &,
              llvm::json::Path);
llvm::json::Value toJSON(const PlaintextScalarEncoding &);
static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                            PlaintextScalarEncoding e) {
  return OS << llvm::formatv("{0:2}", toJSON(e));
}

/// Represents the encoding of a builtin index type.
struct IndexScalarEncoding {};
bool fromJSON(const llvm::json::Value, IndexScalarEncoding &, llvm::json::Path);
llvm::json::Value toJSON(const IndexScalarEncoding &);
static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                            IndexScalarEncoding e) {
  return OS << llvm::formatv("{0:2}", toJSON(e));
}

/// Represents the encoding of a scalar value.
using ScalarEncoding = std::variant<
    EncryptedIntegerScalarEncoding, EncryptedChunkedIntegerScalarEncoding,
    EncryptedBoolScalarEncoding, PlaintextScalarEncoding, IndexScalarEncoding>;
bool fromJSON(const llvm::json::Value, ScalarEncoding &, llvm::json::Path);
llvm::json::Value toJSON(const ScalarEncoding &);
static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                            ScalarEncoding e) {
  return OS << llvm::formatv("{0:2}", toJSON(e));
}

/// Represents the encoding of a tensor value.
struct TensorEncoding {
  ScalarEncoding scalarEncoding;
};
bool fromJSON(const llvm::json::Value, TensorEncoding &, llvm::json::Path);
llvm::json::Value toJSON(const TensorEncoding &);
static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                            TensorEncoding e) {
  return OS << llvm::formatv("{0:2}", toJSON(e));
}

/// Represents the encoding of either an input or output value of a circuit.
using Encoding = std::variant<TensorEncoding, ScalarEncoding>;
bool fromJSON(const llvm::json::Value, Encoding &, llvm::json::Path);
llvm::json::Value toJSON(const Encoding &);
static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, Encoding e) {
  return OS << llvm::formatv("{0:2}", toJSON(e));
}

/// Represents the encodings of a circuit.
struct CircuitEncodings {
  std::vector<Encoding> inputEncodings;
  std::vector<Encoding> outputEncodings;
};
bool fromJSON(const llvm::json::Value, CircuitEncodings &, llvm::json::Path);
llvm::json::Value toJSON(const CircuitEncodings &);
static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                            CircuitEncodings e) {
  return OS << llvm::formatv("{0:2}", toJSON(e));
}

llvm::Expected<CircuitEncodings> getCircuitEncodings(
    llvm::StringRef functionName, mlir::ModuleOp module,
    std::optional<::concretelang::clientlib::ChunkInfo> maybeChunkInfo);

} // namespace encodings
} // namespace concretelang
} // namespace mlir

#endif
