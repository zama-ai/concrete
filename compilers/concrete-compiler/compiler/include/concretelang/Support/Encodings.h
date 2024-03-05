// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
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

#include "capnp/message.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"

using concretelang::protocol::Message;

namespace mlir {
namespace concretelang {
namespace encodings {

llvm::Expected<Message<concreteprotocol::ProgramEncodingInfo>>
getProgramEncoding(mlir::ModuleOp module);

void setProgramEncodingModes(
    Message<concreteprotocol::ProgramEncodingInfo> &info,
    std::optional<
        Message<concreteprotocol::IntegerCiphertextEncodingInfo::ChunkedMode>>
        maybeChunk,
    std::optional<V0FHEContext> maybeFheContext);

} // namespace encodings
} // namespace concretelang
} // namespace mlir

#endif
