// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_PROGRAMINFOGENERATION_H_
#define CONCRETELANG_SUPPORT_PROGRAMINFOGENERATION_H_

#include <memory>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinOps.h>

#include "concrete-protocol.pb.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/Support/Encodings.h"
#include "concretelang/Support/V0Parameters.h"

namespace mlir {
namespace concretelang {

using ::concretelang::clientlib::ChunkInfo;
using ::concretelang::clientlib::ClientParameters;

llvm::Expected<concreteprotocol::ProgramInfo>
createProgramInfoFromTFHE(mlir::ModuleOp module, llvm::StringRef functionName,
                          int bitsOfSecurity,
                          concreteprotocol::CircuitEncodingInfo &encodings);

} // namespace concretelang
} // namespace mlir

#endif
