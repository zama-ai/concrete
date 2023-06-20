// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_ENCODINGS_H_
#define CONCRETELANG_SUPPORT_ENCODINGS_H_

#include <map>
#include <memory>
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

#include "concrete-protocol.pb.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"

namespace mlir {
namespace concretelang {
namespace encodings {

llvm::Expected<concreteprotocol::CircuitEncodingInfo>
getCircuitEncodings(llvm::StringRef functionName, mlir::ModuleOp module);

void setCircuitEncodingModes(
    concreteprotocol::CircuitEncodingInfo &info,
    std::optional<concreteprotocol::IntegerCiphertextEncodingInfo::ChunkedMode>
        maybeChunk,
    std::optional<V0FHEContext> maybeFheContext);

} // namespace encodings
} // namespace concretelang
} // namespace mlir

#endif
