// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_PROGRAMINFOGENERATION_H_
#define CONCRETELANG_SUPPORT_PROGRAMINFOGENERATION_H_

#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Support/Encodings.h"
#include "concretelang/Support/V0Parameters.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>

using concretelang::protocol::Message;

namespace mlir {
namespace concretelang {

llvm::Expected<Message<concreteprotocol::ProgramInfo>>
createProgramInfoFromTfheDialect(
    mlir::ModuleOp module, int bitsOfSecurity,
    const Message<concreteprotocol::ProgramEncodingInfo> &encodings,
    bool compressEvaluationKeys, bool compressInputCiphertexts,
    concrete_optimizer::PublicKey withPublicKeys);

} // namespace concretelang
} // namespace mlir

#endif
