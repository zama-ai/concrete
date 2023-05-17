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
#include "concrete-protocol.pb.h"

namespace protocol = concreteprotocol;

namespace mlir {
namespace concretelang {
namespace encodings {

/// Represents the encodings of a circuit.

llvm::Expected<std::unique_ptr<protocol::CircuitEncodingInfo>> getCircuitEncodings(
    llvm::StringRef functionName, mlir::ModuleOp module,
    std::optional<::concretelang::clientlib::ChunkInfo> maybeChunkInfo);

} // namespace encodings
} // namespace concretelang
} // namespace mlir

#endif
